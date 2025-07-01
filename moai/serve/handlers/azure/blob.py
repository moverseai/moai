import glob
import logging
import os
import typing
import zipfile
from collections.abc import Callable

from azure.storage.blob import BlobServiceClient

from moai.core.execution.common import _create_accessor

log = logging.getLogger(__name__)

__all__ = [
    "AzureBlobInputHandler",
    "AzureZipInputHandler",
    "AzureBlobOutputHandler",
    "AzureZipOutputHandler",
]


class AzureZipInputHandler(Callable):
    def __init__(
        self,
        connection_string: str,  # alias to retrieve connection string from json
        container_name: str,  # name of the container to download data from
        blob_paths: typing.List[str],  # keys to extract resources from json
        json_key: str,  # key to extract working dir from incoming json
        alias: typing.List[str],  # keys of files to be saved
    ):
        """
        Responsible for downlowding data from Azure Blob Storage.

        Args:
            connection_string (str): Connection string for Azure Blob Storage (already set as environmental variable).
            container_name (str): Name of the container to download data from.
            blob_paths (typing.List[str]): Keys to extract resources from json.
            json_key (str): Key to extract working dir from incoming json.
            alias (typing.List[str]): Names of files to be saved.
        """
        # self.blob_service_client = BlobServiceClient.from_connection_string(
        #     connection_string,
        # )
        self.connection_string = connection_string
        self.container_name = container_name
        self.json_key = json_key
        self.blob_paths = blob_paths
        self.blob_acecessors = [_create_accessor(bl_path) for bl_path in blob_paths]
        self.alias_accessors = [
            _create_accessor(al) for al in alias
        ]  # to parse from input json
        log.info(
            "Azure Blob Storage connection established to container: %s", container_name
        )

    def __call__(
        self, json: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:

        if self.json_key not in json:
            log.error(f"json key: {self.json_key}, not found in json request")
        working_dir = json[self.json_key]

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
        for bl_acc, al_acc in zip(self.blob_acecessors, self.alias_accessors):
            blob_name = bl_acc(json)
            blob_client = blob_service_client.get_blob_client(
                container=container, blob=blob_name
            )
            al = al_acc(json)
            download_file_path = os.path.join(working_dir, al)
            # create dir if not exists
            os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            # check if download file is .zip or not and extract if it is
            if download_file_path.endswith(".zip"):
                # get filename without extension
                folder_name = os.path.splitext(download_file_path)[0]
                log.info(f"Extracting zip file to {folder_name}")
                with zipfile.ZipFile(download_file_path, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(working_dir, folder_name))
            log.info(f"Download blob {blob_name} to {download_file_path}")

        return {"is_success": True, "message": "Data downloaded successfully."}


class AzureBlobInputHandler(Callable):
    def __init__(
        self,
        connection_string: str,  # alias to retrieve connection string from json
        container_name: str,  # name of the container to download data from
        blob_paths: typing.List[str],  # keys to extract resources from json
        json_key: str,  # key to extract working dir from incoming json
        alias: typing.List[str],  # names of files to be saved
    ):
        """
        Responsible for downlowding data from Azure Blob Storage.

        Args:
            connection_string (str): Connection string for Azure Blob Storage (already set as environmental variable).
            container_name (str): Name of the container to download data from.
            blob_paths (typing.List[str]): Keys to extract resources from json.
            json_key (str): Key to extract working dir from incoming json.
            alias (typing.List[str]): Names of files to be saved.
        """
        # self.blob_service_client = BlobServiceClient.from_connection_string(
        #     connection_string,
        # )
        self.connection_string = connection_string
        self.container_name = container_name
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

        if self.json_key not in json:
            log.error(f"json key: {self.json_key}, not found in json request")
        working_dir = json[self.json_key]

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
            download_file_path = os.path.join(working_dir, al)
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

        if self.json_key not in input_json:
            log.error(f"json key: {self.json_key}, not found in json request")
        working_dir = input_json[self.json_key]

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
            local_file = os.path.join(working_dir, al)
            # Create a blob client using the local file name as the name for the blob
            blob_client = blob_service_client.get_blob_client(
                container=container, blob=upload_file_path
            )
            # Upload the created file
            with open(file=local_file, mode="rb") as data:
                blob_client.upload_blob(data, overwrite=self.overwrite)

            log.debug(f"Upload file {al} to {upload_file_path}")

        return [{"is_success": True, "message": "Data uploaded successfully."}]


class AzureZipOutputHandler(AzureBlobOutputHandler):
    def __init__(
        self,
        connection_string: str,
        container_name: str,
        blob_paths: typing.List[str],  # keys to extract resources from json
        alias: typing.List[str],  # names of files to be uploaded without suffix
        json_key: str,
        multi_actors: str,  # key to extract resources from input json
        suffix_key: str = "suffix",  # key to extract suffix of local files
        overwrite: bool = True,  # overwrite existing files
    ):
        """
        Handles uploading multiple files as a zip archive to Azure Blob Storage.

        This handler extends AzureBlobOutputHandler to provide functionality for
        zipping multiple files before uploading them to Azure Blob Storage. It extracts
        resources from JSON input, creates a zip archive, and uploads it to the specified
        Azure Blob container.

        Args:
            connection_string (str): Azure Storage account connection string
            container_name (str): Name of the Azure Blob container
            blob_paths (List[str]): List of keys to extract resources from JSON input
            alias (List[str]): List of filenames to use when uploading the files
            json_key (str): Key to extract data from input JSON
            multi_actors (str): Key to extract multiple resources from input JSON
            overwrite (bool, optional): Whether to overwrite existing blobs. Defaults to True.

        Example:
            ```python
            handler = AzureZipHandler(
                connection_string="DefaultEndpointsProtocol=https;AccountName=...",
                container_name="my-container",
                blob_paths=["output1", "output2"],
                alias=["file1", "file2"],
                json_key="results",
                multi_actors="batch_output",
                overwrite=True
            )
            ```
        """
        super().__init__(
            connection_string, container_name, blob_paths, alias, json_key, overwrite
        )
        self.multi_actors = multi_actors
        self.suffix_key = suffix_key

    def __call__(
        self, json: typing.Mapping[str, typing.Any], void: typing.Any
    ) -> typing.Any:
        input_json = void[0].get("body") or void[0].get("raw")
        if self.multi_actors not in input_json:
            log.error(
                f"json key: {self.multi_actors}, not found in json request, reverting to default"
            )
            multi_actors = False
        else:
            log.info(f"Found key: {self.multi_actors} in json request")
            multi_actors = eval(input_json[self.multi_actors])
            log.info(f"Found multiple actors: {multi_actors}")
        working_dir = input_json[self.json_key]
        suffix = input_json[self.suffix_key]
        original_alias = self.alias.copy()
        # if multi_actors:
        #     # zip all files that need to be uploaded into one file and upload this
        #     for al in self.alias:
        #         # find suffix of al file
        #         new_alias = []
        #         # get all files from local dir with the same suffix
        #         local_files = glob.glob(os.path.join(working_dir, f"*.{suffix}"))
        #         log.info(f"Zipping files: {local_files} with suffix: {suffix}")
        #         # replace suffix with zip
        #         al = al.replace(suffix, "zip")
        #         # zip all files
        #         zip_file = os.path.join(working_dir, al)
        #         with zipfile.ZipFile(zip_file, "w") as zipf:
        #             for path_file in local_files:
        #                 log.info(f"Adding {path_file} to zip file")
        #                 # get only filename from f path
        #                 file_name = os.path.basename(path_file)
        #                 zipf.write(path_file, arcname=file_name)
        #         log.info(f"Zip file created: {zip_file} and new alias: {al}")
        #         new_alias.append(al)
        #     self.alias = new_alias
        if multi_actors:
            # zip all files that need to be uploaded into one file and upload this
            for al in self.alias:
                # find suffix of al file
                new_alias = []
                # get all files from local dir with the same suffix
                local_files = glob.glob(os.path.join(working_dir, f"*.{suffix}"))
                log.info(f"Zipping files: {local_files} with suffix: {suffix}")
                # add .zip suffix to al
                al = f"{al}.zip"
                # zip all files
                zip_file = os.path.join(working_dir, al)
                with zipfile.ZipFile(zip_file, "w") as zipf:
                    for path_file in local_files:
                        log.info(f"Adding {path_file} to zip file")
                        # get only filename from f path
                        file_name = os.path.basename(path_file)
                        zipf.write(path_file, arcname=file_name)
                log.info(f"Zip file created: {zip_file} and new alias: {al}")
                new_alias.append(al)
        else:
            # update suffix of alias
            new_alias = []
            for al in self.alias:
                # add suffix to al
                al = f"{al}.{suffix}"  # TODO: support different suffixes for each of local files
                new_alias.append(al)
        self.alias = new_alias
        # call parent class to upload zip file
        log.info(f"Calling parent class with alias: {self.alias}")
        try:
            super().__call__(json, void)
        except Exception as e:
            log.error(f"An error has occured while uploading zip file:\n{e}")
            log.info(f"Reverting back to original alias: {original_alias}")
            self.alias = original_alias

        # revert back to original alias
        log.info(f"Reverting back to original alias: {original_alias}")
        self.alias = original_alias
        log.info(f"Reverting back to original alias: {self.alias}")

        return [{"is_success": True, "message": "Zip handler succeded."}]
