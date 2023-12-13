from collections.abc import Callable
import typing
import logging
from azure.storage.blob import BlobServiceClient
import os
from moai.monads.execution.cascade import _create_accessor


log = logging.getLogger(__name__)


class AzureBlobInputHandler(Callable):
    def __init__(
        self,
        connection_string: str,
        container_name: str,
        blob_paths: typing.List[str],  # keys to extract resources from json
        working_dir: str,  # path to working dir
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
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string,
        )
        self.container_name = container_name
        self.working_dir = working_dir
        self.blob_paths = blob_paths
        self.blob_acecessors = [_create_accessor(bl_path) for bl_path in blob_paths]
        self.alias = alias
        log.info(
            "Azure Blob Storage connection established to container: %s", container_name
        )

    def __call__(
        self, json: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        for bl_acc, al in zip(self.blob_acecessors, self.alias):
            blob_name = bl_acc(json)
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
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
    ):
        """
        Responsible for uploading data to Azure Blob Storage.

        Args:
            connection_string (str): Connection string for Azure Blob Storage (already set as environmental variable).
            container_name (str): Name of the container to upload data to.
            blob_paths (typing.List[str]): Keys to extract resources from json.
            working_dir (str): Path to working dir.
            alias (typing.List[str]): Names of files to be uploaded.

        """
        # connect_str = os.environ[connection_string]
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string,
        )
        self.blob_paths = blob_paths
        self.blob_acecessors = [_create_accessor(bl_path) for bl_path in blob_paths]
        self.working_dir = working_dir
        self.alias = alias
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
        for bl_acc, al in zip(self.blob_acecessors, self.alias):
            log.debug(f"Uploading {al} to Azure Blob Storage...")
            log.debug(f"blob path: {bl_acc(input_json)}")
            upload_file_path = bl_acc(input_json)
            local_file = os.path.join(self.working_dir, al)
            # Create a blob client using the local file name as the name for the blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=upload_file_path
            )
            # Upload the created file
            with open(file=local_file, mode="rb") as data:
                blob_client.upload_blob(data)

            log.debug(f"Upload file {al} to {upload_file_path}")

        return [{"is_success": True, "message": "Data uploaded successfully."}]
