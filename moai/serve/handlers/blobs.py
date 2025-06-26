import glob
import json as js
import logging
import os
import typing
import zipfile
from collections.abc import Callable

from moai.core.execution.common import _create_accessor

log = logging.getLogger(__name__)

try:
    from iopath.common.azure_blob import AzureBlobPathHandler
    from iopath.common.file_io import HTTPURLHandler, PathManager
    from iopath.common.gcs import GcsPathHandler
except ImportError as e:
    log.error(
        "iopath is not installed. Please install it with `pip install moai[all]`."
    )
    raise e


class BlobInputHandler(Callable):
    def __init__(
        self,
        blob_paths: typing.List[str],
        connection_string: str,
        container_name: str,
        json_key: str,
        alias: typing.List[str],
    ) -> None:
        """
        Responsible for downloading blobs from different providers Azure, GCS, HTTP, etc.

        Args:
            connection_string: str to search connection string in json data.
            json_key (str): Key in the json data where the working directory is stored.
            blob_paths (typing.List[str]): List of blob paths to download.
        """

        self.blob_paths = blob_paths
        self.connection_string = connection_string
        self.container_name = container_name
        self.json_key = json_key
        self.alias = alias
        self.blob_acecessors = [_create_accessor(bl_path) for bl_path in blob_paths]
        self.alias_accessors = [
            _create_accessor(al) for al in alias
        ]  # to parse from input json
        self.provider = None
        log.info(
            f"Download Blobs Storage connection handlers instantiated with aliases: {self.alias}, paths: {self.blob_paths} and json key: {self.json_key}",
        )

    def parse_connection_string(self, conn_str):
        """Parse Azure Storage connection string into a dictionary."""
        parts = conn_str.split(";")
        parsed = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key] = value
        return parsed

    def __call__(
        self, json: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        # TODO: we need to check how json is passed actually
        # print(json)
        if self.json_key not in json:
            log.error(f"json key: {self.json_key}, not found in json request")
        working_dir = json[self.json_key]
        # search for connection string from json
        connect_str = json[self.connection_string]
        # check if connection string is not empty
        if not connect_str:
            log.error(f"Connection string {self.connection_string} is empty")
            return [{"is_success": False, "message": "Connection string is empty"}]
        else:
            # log.info(f"Connection string: {connect_str}")
            http_handler = HTTPURLHandler()
            # TODO: Add local option
            path_manager = PathManager()
            path_manager.register_handler(http_handler)
            # we need to check the cloud provider
            if "DefaultEndpointsProtocol" in connect_str:
                # Azure Blob Storage
                log.info("Using Azure Blob Storage")
                path_handler = AzureBlobPathHandler(connection_string=connect_str)
                path_manager.register_handler(path_handler)
                self.provider = "az"
            elif "service_account" == js.loads(connect_str)["type"]:
                # GCS Storage
                path_handler = GcsPathHandler(connection_json=connect_str)
                path_manager.register_handler(path_handler)
                self.provider = "gcs"
            # TODO: add local
            else:
                log.info("Not known cloud provider found trying Local or FTP.")
                self.provider = "ftp"

            container = json[self.container_name]
            for bl_acc, al in zip(self.blob_acecessors, self.alias):
                blob_name = bl_acc(json)
                download_file_path = os.path.join(working_dir, al)
                # create dir if not exists
                os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
                if self.provider == "az":
                    parsed = self.parse_connection_string(connect_str)
                    account = parsed.get("AccountName")
                    uri = f"blob://{account}/{container}/{blob_name}"
                elif self.provider == "gcs":
                    uri = f"gs://{container}/{blob_name}"
                else:
                    # means ftp, pass all the link
                    uri = blob_name
                path_manager.copy(uri, download_file_path, overwrite=True)
                log.info(f"Download blob {blob_name} to {download_file_path}")

        return {"is_success": True, "message": "Data downloaded successfully."}


class BlobOutputHandler(Callable):
    def __init__(
        self,
        blob_paths: typing.List[str],
        connection_string: str,
        container_name: str,
        json_key: str,
        alias: typing.List[str],
        overwrite: bool = True,  # overwrite existing files
    ) -> None:
        """
        Responsible for downloading blobs from different providers Azure, GCS, HTTP, etc.

        Args:
            connection_string: str to search connection string in json data.
            json_key (str): Key in the json data where the working directory is stored.
            blob_paths (typing.List[str]): List of blob paths to download.
        """

        self.blob_paths = blob_paths
        self.connection_string = connection_string
        self.container_name = container_name
        self.json_key = json_key
        self.alias = alias
        self.blob_acecessors = [_create_accessor(bl_path) for bl_path in blob_paths]
        self.provider = None
        self.overwrite = overwrite
        log.info(
            f"Upload Blobs Storage connection handlers instantiated with aliases: {self.alias}, paths: {self.blob_paths} and json key: {self.json_key}",
        )

    def parse_connection_string(self, conn_str):
        """Parse Azure Storage connection string into a dictionary."""
        parts = conn_str.split(";")
        parsed = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key] = value
        return parsed

    def __call__(
        self, json: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        # TODO: we need to check how json is passed actually
        # log.info(f"Input json for Output Blob Handler: {json}")
        input_json = void[0].get("body") or void[0].get("raw")
        if self.json_key not in input_json:
            log.error(f"json key: {self.json_key}, not found in json request")
        working_dir = input_json[self.json_key]
        # search for connection string from json
        connect_str = input_json[self.connection_string]
        # check if connection string is not empty
        if not connect_str:
            log.error(f"Connection string {self.connection_string} is empty")
            return [{"is_success": False, "message": "Connection string is empty"}]
        else:
            http_handler = HTTPURLHandler()
            # TODO: Add local option
            path_manager = PathManager()
            path_manager.register_handler(http_handler)
            # we need to check the cloud provider
            if "DefaultEndpointsProtocol" in connect_str:
                # Azure Blob Storage
                log.info("Using Azure Blob Storage")
                path_handler = AzureBlobPathHandler(connection_string=connect_str)
                path_manager.register_handler(path_handler)
                self.provider = "az"
            elif "service_account" == js.loads(connect_str)["type"]:
                # GCS Storage
                path_handler = GcsPathHandler(connection_json=connect_str)
                path_manager.register_handler(path_handler)
                self.provider = "gcs"
            # TODO: add local
            else:
                log.info("Not known cloud provider found trying Local or FTP.")
                self.provider = "ftp"

            container = json[self.container_name]
            for bl_acc, al in zip(self.blob_acecessors, self.alias):
                upload_file_path = bl_acc(input_json)
                local_file = os.path.join(working_dir, al)
                # assert local file exists
                if not os.path.isfile(local_file):
                    log.error(f"Local file {local_file} does not exist.")
                    return [
                        {
                            "is_success": False,
                            "message": f"Local file {local_file} does not exist.",
                        }
                    ]
                if self.provider == "az":
                    parsed = self.parse_connection_string(connect_str)
                    account = parsed.get("AccountName")
                    uri = f"blob://{account}/{container}/{upload_file_path}"
                elif self.provider == "gcs":
                    uri = f"gs://{container}/{upload_file_path}"
                else:
                    # means ftp, pass all the link
                    uri = upload_file_path
                path_manager.copy(local_file, uri, overwrite=self.overwrite)
                log.info(f"Uploaded blob {al} to {upload_file_path}")

        return [{"is_success": True, "message": "Data uploaded successfully."}]
