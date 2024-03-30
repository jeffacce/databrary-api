import os
import time
import json
import sheet
import requests
import functools
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from difflib import get_close_matches
from typing import Dict, List, IO, Optional, Union

# https://github.com/databrary/databrary/blob/3d72b67991df39ddb4b0930386ea94e2c5bb5733/web/service/upload.coffee#L14
CHUNK_SIZE = 1048576
THREADS = 1
ERROR_STATUS_CODES = [400, 403, 404, 409, 415, 500, 501]
SUCCESS_STATUS_CODES = [200, 201, 202, 204]
RETRY_LIMIT = 3
RETRY_WAIT_TIME = 1  # seconds
RETRY_BACKOFF = 2  # exponential backoff

# === ENDPOINTS ===
LOGIN = "https://nyu.databrary.org/api/user/login"
GET_VOL_BY_ID = "https://nyu.databrary.org/api/volume/{volumeid}?access&citation&links&funding&top&tags&excerpts&comments&records&containers=all&metrics&state"
CREATE_SLOT = "https://nyu.databrary.org/api/volume/{volumeid}/slot"
CREATE_UPLOAD_FLOW = "https://nyu.databrary.org/api/volume/{volumeid}/upload"
UPLOAD_CHUNK = "https://nyu.databrary.org/api/upload"
CREATE_FILE_FROM_FLOW = "https://nyu.databrary.org/api/volume/{volumeid}/asset"
UPDATE_SLOT = "https://nyu.databrary.org/api/slot/{slotid}"
QUERY_SLOT = "https://nyu.databrary.org/api/slot/{slotid}/-?records&assets&excerpts&tags&comments"
DOWNLOAD_ASSET = (
    "https://nyu.databrary.org/slot/{slotid}/asset/{assetid}/download?inline=false"
)
DOWNLOAD_SESSION_ZIP = (
    "https://nyu.databrary.org/volume/{volumeid}/slot/{slotid}/zip/{original}"
)
CREATE_PARTICIPANT = "https://nyu.databrary.org/api/volume/{volumeid}/record"
GET_PARTICIPANT_BY_ID = "https://nyu.databrary.org/api/record/{recordid}"
UPDATE_METRIC = "https://nyu.databrary.org/api/record/{recordid}/metric/{metricid}"


def chunkify(filesize: int, chunk_size: int) -> List[int]:
    chunk_sizes = [chunk_size] * (filesize // chunk_size)
    if filesize % chunk_size > 0:
        chunk_sizes.append(filesize % chunk_size)
    return chunk_sizes


class Volume:
    def __init__(
        self,
        client: "Client",
        metadata: Dict,
    ):
        self._client = client
        self.metadata = metadata
        self.id_ = metadata["id"]

    def get_session_by_id(self, session_id: int) -> "Session":
        if session_id not in [session["id"] for session in self.metadata["containers"]]:
            raise ValueError(f"Session {session_id} not found.")
        else:
            resp = self._client.s.get(QUERY_SLOT.format(slotid=session_id)).json()
            return Session(self._client, self, resp)

    def get_session_by_name(self, name: str) -> Union["Session", None]:
        ids = [
            session["id"]
            for session in self.metadata["containers"]
            if session.get("name") == name
        ]
        if len(ids) == 0:
            return None
        if len(ids) > 1:
            raise ValueError(f"Multiple sessions with name {name} found.")
        return self.get_session_by_id(ids[0])

    @property
    def _session_ids(self) -> Dict:
        result = {
            session["id"]: session.get("name", "")
            for session in self.metadata["containers"]
            if session.get("top") is None
        }
        result = pd.DataFrame(
            {
                "id": list(result.keys()),
                "name": list(result.values()),
            }
        )
        return result

    @property
    def records(self) -> pd.DataFrame:
        sessions = filter(
            lambda x: x["id"] in self._session_ids["id"].values,
            self.metadata["containers"],
        )
        result = pd.concat(
            [
                sheet.build_session_df_row(session, self.metadata["records"])
                for session in sessions
            ],
            axis=0,
        )
        result = result.reset_index(drop=True)
        return result
    
    @property
    def participants(self) -> pd.DataFrame:
        participants = filter(
            lambda x: x["category"] == 1,
            self.metadata["records"],
        )
        result = pd.concat(
            [
                sheet.build_record_df(participant)
                for participant in participants
            ],
            axis=0,
        )
        result = result.reset_index(drop=True)
        return result

    def create_session(self) -> "Session":
        # TODO: refactor csrf + requests.session to Context
        resp = self._client.s.post(
            CREATE_SLOT.format(volumeid=self.metadata["id"]),
            json={"csverf": self._client.csrf_token},
        )
        session_id = resp.json()["id"]
        self.refresh()
        return self.get_session_by_id(session_id)

    def refresh(self) -> "Volume":
        volume = self._client.get_volume_by_id(self.id_)
        self.metadata = volume.metadata
        return self
    
    def get_participant_by_id(self, participant_id: str) -> "Participant":
        participant_id = str(participant_id)  # cast to str in case of int ids
        participant_metadata = filter(
            lambda x: x["category"] == 1,
            self.metadata["records"],
        )
        matches = list(filter(lambda x: x["measures"]["1"] == participant_id, participant_metadata))
        if len(matches) == 0:
            raise ValueError(f"Participant {participant_id} not found.")
        elif len(matches) > 1:
            raise ValueError(f"Multiple participants with ID {participant_id} found.")
        return self._get_participant_by_api_id(matches[0]["id"])
    
    def _get_participant_by_api_id(self, participant_id: int) -> "Participant":
        resp = self._client.s.get(GET_PARTICIPANT_BY_ID.format(recordid=participant_id))
        return Participant(self._client, self, resp.json())
    
    def _create_participant(self) -> "Participant":
        resp = self._client.s.post(
            CREATE_PARTICIPANT.format(volumeid=self.metadata["id"]),
            json={"csverf": self._client.csrf_token, "category": 1},
        )
        return self._get_participant_by_api_id(resp.json()["id"])
    
    def create_participant_with_id(self, participant_id: str) -> "Participant":
        participant = self._create_participant()
        participant.update(ID=participant_id)
        return participant
    
    # TODO: update metric
    # def update_metric(metric_id, datum):

    def __str__(self):
        return f"Volume {self.id_}: {self.metadata['name']}"

    def __repr__(self):
        return f"Volume {self.id_}: {self.metadata['name']}"


class Participant:
    def __init__(
        self,
        client: "Client",
        volume: "Volume",
        metadata: Dict,
    ):
        self._client = client
        self.volume = volume
        self.metadata = metadata
        self.id_ = metadata["id"]  # this is the internal API id, not the visible participant id
    
    def _update_metric(self, metric_id: int, datum):
        assert metric_id in sheet.PARTICIPANT_METRICS, f"Metric {metric_id} does not exist for Participant. This is likely a bug; please report this issue."
        resp = self._client.s.post(
            UPDATE_METRIC.format(recordid=self.id_, metricid=metric_id),
            json={"csverf": self._client.csrf_token, "datum": datum},
        )
        return resp.status_code
    
    def refresh(self):
        participant = self.volume._get_participant_by_api_id(self.id_)
        self.metadata = participant.metadata
        return self
    
    def update(self, **kwargs):
        """
        Updates participant information. The header must be included in the volume spreadsheet, and be one of the following:
            - `id`
            - `info`
            - `description`
            - `birthdate`
            - `gender`
            - `race`
            - `ethnicity`
            - `gestational_age`
            - `pregnancy_term`
            - `birth_weight`
            - `disability`
            - `language`
            - `country`
            - `state`
            - `setting` (deprecated)
        
        Example:
        ```
        participant.update(id="123", info="some info", description="some description")
        ```

        Returns:
            - Participant: the updated participant object
        """
        payload = {}
        for k, v in kwargs.items():
            k_slug = sheet.slugify(k)
            if k_slug not in sheet.PARTICIPANT_METRICS_STOI:
                closest_match = get_close_matches(k_slug, sheet.PARTICIPANT_METRICS_STOI.keys(), n=1)[0]
                raise ValueError(f"{k} doesn't seem to be a valid participant header. Did you mean {closest_match}?")
            else:
                payload[sheet.PARTICIPANT_METRICS_STOI[k_slug]] = v
        not_in_spreadsheet = [sheet.PARTICIPANT_METRICS[k] for k in payload.keys() if k not in self.volume.metadata["metrics"]]
        if len(not_in_spreadsheet) > 0:
            raise ValueError(f"The following headers are not part of the spreadsheet: {not_in_spreadsheet}")
        for k, v in payload.items():
            self._update_metric(k, v)
        return self.refresh()
    
    @property
    def records(self) -> pd.DataFrame:
        return sheet.build_record_df(self.metadata)

    def __str__(self) -> str:
        return f"Participant {self.metadata['measures']['1']}"
    
    def __repr__(self) -> str:
        return f"Participant {self.metadata['measures']['1']}"

class Session:
    def __init__(
        self,
        client: "Client",
        volume: "Volume",
        metadata: Dict,
    ):
        self._client = client
        self.volume = volume
        self.metadata = metadata
        self.id_ = metadata["id"]

    @property
    def filenames(self) -> List[str]:
        return [asset["name"] for asset in self.metadata["assets"]]

    def update(self, name: Optional[str] = None) -> int:
        json = {"csverf": self._client.csrf_token}
        if name is not None:
            json["name"] = name
        resp = self._client.s.post(
            UPDATE_SLOT.format(slotid=self.metadata["id"]), json=json
        )
        return resp.status_code

    def upload_file(
        self,
        filepath: Union[str, os.PathLike],
        upload_name: Optional[str] = None,
    ):
        upload_name = upload_name or os.path.basename(filepath)
        filesize = os.path.getsize(filepath)
        upload_flow_id = self._client._create_upload_flow(
            self.volume.id_, upload_name, filesize
        )

        chunk_sizes = chunkify(filesize, CHUNK_SIZE)

        f = open(filepath, "rb")
        pbar = tqdm(
            total=filesize,
            desc="Uploading",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )

        def callback(chunk_size, _):
            pbar.update(chunk_size)
            pbar.refresh()

        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = []
            for i in range(len(chunk_sizes)):
                this_future = executor.submit(
                    self._client._upload_chunk_with_retry,
                    f,
                    chunk_sizes,
                    i,
                    upload_flow_id,
                    upload_name,
                )
                this_future.add_done_callback(
                    functools.partial(callback, chunk_sizes[i])
                )
                futures.append(this_future)
            for future in concurrent.futures.as_completed(futures):
                success = future.result()
                if not success:
                    for elem in futures:
                        elem.cancel()
                    pbar.set_description_str("Error")
                    pbar.close()
                    f.close()
                    print(
                        f"There was an error uploading the file: {filepath}. Upload failed."
                    )
                    return False
        pbar.update(1)  # deliberately overflow filesize to collapse the pbar
        pbar.set_description_str("Done")
        pbar.close()
        f.close()

        self._client._create_file_from_flow(
            upload_flow_id, self.volume.id_, self.id_, upload_name
        )
        return True

    def download_file_by_id(
        self, asset_id: int, filepath: Optional[os.PathLike] = None
    ):
        asset = list(filter(lambda x: x["id"] == asset_id, self.metadata["assets"]))
        assert len(asset) > 0, f"Asset {asset_id} not found."
        resp = self._client.s.get(
            DOWNLOAD_ASSET.format(slotid=self.id_, start=0, end=0, assetid=asset_id),
            stream=True,
        )
        filename = json.loads(resp.headers["content-disposition"].split("=")[1])
        filesize = int(resp.headers["Content-Length"])
        filepath = filepath or filename
        with open(filepath, "wb") as f:
            pbar = tqdm(
                total=filesize,
                desc="Downloading",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.update(1)  # deliberately overflow filesize to collapse the pbar
            pbar.set_description_str("Done")
            pbar.close()

    def download_file_by_name(
        self, filename: str, filepath: Optional[os.PathLike] = None
    ):
        # find ID of asset
        asset = list(filter(lambda x: x["name"] == filename, self.metadata["assets"]))
        assert len(asset) > 0, f"Asset {filename} not found."
        assert len(asset) == 1, f"Multiple assets with name {filename} found."
        asset_id = asset[0]["id"]
        self.download_file_by_id(asset_id, filepath)

    def download_zip(
        self, filepath: Optional[os.PathLike] = None, original: bool = False
    ):
        resp = self._client.s.get(
            DOWNLOAD_SESSION_ZIP.format(
                volumeid=self.volume.id_, slotid=self.id_, original=original
            )
        )
        filename = json.loads(resp.headers["content-disposition"].split("=")[1])
        filesize = int(resp.headers["Content-Length"])
        filepath = filepath or filename
        with open(filepath, "wb") as f:
            pbar = tqdm(
                total=filesize,
                desc="Downloading",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.update(1)

    @property
    def records(self) -> pd.DataFrame:
        return sheet.build_session_df_row(
            self.metadata, self.volume.metadata["records"]
        )

    def __str__(self):
        return f"Session {self.id_}: {self.metadata.get('name', '')}"

    def __repr__(self):
        return f"Session {self.id_}: {self.metadata.get('name', '')}"


class Client:
    def __init__(self, email: str, password: str):
        self.s = requests.session()
        resp = self.s.post(LOGIN, json={"email": email, "password": password})
        self.csrf_token = resp.json()["csverf"]
        self.session_id = resp.cookies["session"]

    def get_volume_by_id(self, volume_id: int) -> Volume:
        resp = self.s.get(GET_VOL_BY_ID.format(volumeid=volume_id))
        return Volume(self, resp.json())

    def _create_upload_flow(
        self, volume_id: int, upload_name: str, filesize: int
    ) -> str:
        resp = self.s.post(
            CREATE_UPLOAD_FLOW.format(volumeid=volume_id),
            json={"filename": upload_name, "size": filesize},
        )
        return resp.text

    def _upload_chunk(
        self,
        f: IO,
        chunk_sizes: List[int],
        chunk_idx: int,
        upload_flow_id: str,
        upload_name: str,
    ) -> bool:
        # seek read for concurrency
        f.seek(chunk_idx * CHUNK_SIZE)
        buf = f.read(CHUNK_SIZE)
        params = {
            "flowChunkNumber": chunk_idx + 1,
            "flowChunkSize": CHUNK_SIZE,
            "flowCurrentChunkSize": chunk_sizes[chunk_idx],
            "flowTotalSize": sum(chunk_sizes),
            "flowIdentifier": upload_flow_id,
            "flowFilename": upload_name,
            "flowRelativePath": upload_name,
            "flowTotalChunks": len(chunk_sizes),
            "csverf": self.csrf_token,
        }
        resp = self.s.post(UPLOAD_CHUNK, params=params, data=buf)
        return resp.status_code in SUCCESS_STATUS_CODES

    def _upload_chunk_with_retry(
        self,
        f: IO,
        chunk_sizes: List[int],
        chunk_idx: int,
        upload_flow_id: str,
        upload_name: str,
    ) -> bool:
        for i in range(RETRY_LIMIT):
            try:
                success = self._upload_chunk(
                    f, chunk_sizes, chunk_idx, upload_flow_id, upload_name
                )
                if success:
                    return True
                else:
                    print(f"Upload failed for {chunk_idx} on attempt {i+1}.")
            except Exception as e:
                print(f"Upload failed for {chunk_idx} on attempt {i+1}. Error: {e}")
                time.sleep(RETRY_WAIT_TIME * RETRY_BACKOFF**i)
        return False

    def _create_file_from_flow(
        self,
        upload_flow_id: str,
        volume_id: int,
        session_id: int,
        upload_name: str,
    ):
        resp = self.s.post(
            CREATE_FILE_FROM_FLOW.format(volumeid=volume_id),
            json={
                "container": session_id,
                "name": upload_name,
                "upload": upload_flow_id,
                "csverf": self.csrf_token,
            },
        )
        return resp.status_code


if __name__ == "__main__":
    VOLUME_ID = 12345
    client = Client(email="john.doe@nyu.edu", password="foobar")  # log into Databrary
    volume = client.get_volume_by_id(VOLUME_ID)  # get a volume
    print(volume.records)  # get a dataframe of all session records in the volume
    session = volume.get_session_by_id(12345)  # get a session
    session = volume.get_session_by_name("Session 1")  # get a session
    print(session.records)  # get a dataframe of session records
    session = volume.create_session()  # create a session in the volume
    session.upload_file("/path/to/file.mp4")  # upload a file to the session
    session.download_file_by_name(
        "file.mp4", "/path/to/download.mp4"
    )  # download a file from the session
    session.download_zip("/path/to/download.zip")  # download a zip of the session
