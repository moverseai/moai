import logging
import os
import typing
from collections.abc import Callable

from azure.storage.blob import BlobServiceClient

from moai.core.execution.common import _create_accessor

log = logging.getLogger(__name__)


class AzureBlobInputHandler(Callable):
    def __init__(
        self,
        connection_string: str,  # alias to retrieve connection string from json
        container_name: str,  # name of the container to download data from
        blob_paths: typing.List[str],  # keys to extract resources from json
        working_dir: str,  # path to working dir
        json_key: str,
        alias: typing.List[str],  # names of files to be saved
    ):
        """
        Responsible for downlowding data from Azure Blob Storage.

        Args:
            connection_string (str): Connection string for Azure Blob Storage (already set as environmental variable).
            container_name (str): Name of the container to download data from.
            blob_paths (typing.List[str]): Keys to extract resources from json.
            working_dir (str): Path to working dir.
            alias (typing.List[str]): Names of files to be saved.
        """
        # self.blob_service_client = BlobServiceClient.from_connection_string(
        #     connection_string,
        # )
        self.connection_string = connection_string
        self.container_name = container_name
        self.working_dir = working_dir
        self.json_key = json_key
        self.blob_paths = blob_paths
        self.blob_acecessors = [_create_accessor(bl_path) for bl_path in blob_paths]
        self.alias = alias
        log.info(
            "Azure Blob Storage connection established to container: %s", container_name
        )

    def __call__(
        self, json: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        if self.working_dir is None:
            self.working_dir = json[self.json_key]
        # initialize connection to Azure Blob Storage
        connect_str = json[self.connection_string]
        try:
            blob_service_client = BlobServiceClient.from_connection_string(
                connect_str,
            )
        except Exception as e:
            log.info(
                f"An error has occured while connecting to Azure Blob Storage:\n{e}"
            )
        container = json[self.container_name]
        for bl_acc, al in zip(self.blob_acecessors, self.alias):
            blob_name = bl_acc(json)
            blob_client = blob_service_client.get_blob_client(
                container=container, blob=blob_name
            )
            download_file_path = os.path.join(self.working_dir, al)
            # create dir if not exists
            os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            log.info(f"Download blob {blob_name} to {download_file_path}")

        return {"is_success": True, "message": "Data downloaded successfully."}


class AzureBlobOutputHandler(Callable):
    def __init__(
        self,
        connection_string: str,
        container_name: str,
        blob_paths: typing.List[str],  # keys to extract resources from json
        working_dir: str,  # path to working dir
        alias: typing.List[str],  # names of files to be uploaded
        json_key: str,
        overwrite: bool = True,  # overwrite existing files
    ):
        """
        Responsible for uploading data to Azure Blob Storage.

        Args:
            connection_string (str): Connection string for Azure Blob Storage (already set as environmental variable).
            container_name (str): Name of the container to upload data to.
            blob_paths (typing.List[str]): Keys to extract resources from json.
            working_dir (str): Path to working dir.
            alias (typing.List[str]): Names of files to be uploaded.
            overwrite (bool, optional): Overwrite existing files. Defaults to True.

        """
        # connect_str = os.environ[connection_string]
        # self.blob_service_client = BlobServiceClient.from_connection_string(
        #     connection_string,
        # )
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_paths = blob_paths
        self.blob_acecessors = [_create_accessor(bl_path) for bl_path in blob_paths]
        self.working_dir = working_dir
        self.json_key = json_key
        self.alias = alias
        self.overwrite = overwrite
        log.info(
            "Azure Blob Upload Storage connection established to container: %s",
            container_name,
        )

    def __call__(
        self, json: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        # NOTE: void is the input json response
        # TODO: need to check batched inference
        input_json = void[0].get("body") or void[0].get("raw")
        if self.working_dir is None:
            self.working_dir = input_json[self.json_key]
        # initialize connection to Azure Blob Storage
        connect_str = input_json[self.connection_string]
        blob_service_client = BlobServiceClient.from_connection_string(
            connect_str,
        )
        container = input_json[self.container_name]
        for bl_acc, al in zip(self.blob_acecessors, self.alias):
            log.debug(f"Uploading {al} to Azure Blob Storage...")
            log.debug(f"blob path: {bl_acc(input_json)}")
            upload_file_path = bl_acc(input_json)
            local_file = os.path.join(self.working_dir, al)
            # Create a blob client using the local file name as the name for the blob
            blob_client = blob_service_client.get_blob_client(
                container=container, blob=upload_file_path
            )
            # Upload the created file
            with open(file=local_file, mode="rb") as data:
                blob_client.upload_blob(data, overwrite=self.overwrite)

            log.debug(f"Upload file {al} to {upload_file_path}")

        return [{"is_success": True, "message": "Data uploaded successfully."}]
