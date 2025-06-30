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
                # path_manager.copy(uri, download_file_path, overwrite=True)
                with path_manager.open(uri, "rb") as blob_file:
                    with open(download_file_path, "wb") as local_file:
                        local_file.write(blob_file.read())
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


class BlobZipInputHandler(Callable):
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
            "Cloud Blob Storage zip input handler initialized with aliases: {}, paths: {}, and json key: {}".format(
                self.alias_accessors, self.blob_acecessors, self.json_key
            )
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

        if self.json_key not in json:
            log.error(f"json key: {self.json_key}, not found in json request")
        working_dir = json[self.json_key]

        # initialize connection to Azure Blob Storage
        connect_str = json[self.connection_string]
        if not connect_str:
            log.error(f"Connection string {self.connection_string} is empty")
            return [{"is_success": False, "message": "Connection string is empty"}]
        else:
            # log.info(f"Connection string: {connect_str}")
            http_handler = HTTPURLHandler()
            # TODO: Add local option
            path_manager = PathManager()
            path_manager.register_handler(http_handler)
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
            else:
                log.info("Not known cloud provider found, trying Local or FTP.")
                self.provider = "ftp"
            # TODO: add local
            container = json[self.container_name]
            for bl_acc, al_acc in zip(self.blob_acecessors, self.alias_accessors):
                blob_name = bl_acc(json)
                al = al_acc(json)
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
                # path_manager.copy(uri, download_file_path, overwrite=True)
                with path_manager.open(uri, "rb") as blob_file:
                    with open(download_file_path, "wb") as local_file:
                        local_file.write(blob_file.read())
                log.info(f"Download blob {blob_name} to {download_file_path}")

        return {"is_success": True, "message": "Data downloaded successfully."}


class BlobZipOutputHandler(BlobOutputHandler):
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

        This handler extends BlobOutputHandler to provide functionality for
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
            handler = BlobZipOutputHandler(
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
            connection_string=connection_string,
            container_name=container_name,
            blob_paths=blob_paths,
            alias=alias,
            json_key=json_key,
            overwrite=overwrite,
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
