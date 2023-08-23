import os
import webbrowser
import threading
import time
import socket
from typing import Dict, Optional, List
import re
import subprocess

# pip install docker fire
import docker
from docker.types import DeviceRequest
from docker.client import DockerClient
import fire


def is_port_free(port: int, host: str = 'localhost') -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

class Run():
    """
    Run docker containers and mount local volumes.
    """
    DEFAULT_PORT=8888

    def __init__(self):
        # executed by fire.Fire when no commands/args are passed
        self.default_function = self._run

        # helps make sure _run() gets executed
        self.launched = False

        # Get the absolute path of the current script
        script_path = os.path.abspath(__file__)
        # Get the directory of the current script
        self.script_dir = os.path.dirname(script_path)

        # set defaults here
        self.container_name = "ml-dev-container"
        self.container_img = "ml-dev-container"
        self.img_tag = ":latest"
        self.environment = {
            "JUPYTER_TOKEN": "passwd",
            # KEY: os.environ.get(KEY),
        }
        self.port_map = {self.DEFAULT_PORT:self.DEFAULT_PORT}  # {container:host}
        self.mount_vols={}
        self._set_default_volumes()

        # for interacting with docker
        self.client = docker.from_env()
        
        self.error = False


    def _get_gpus(
            self
        ) -> List[dict]:
        if not self._host_has_gpu():
            return []
        return [DeviceRequest(count=-1, capabilities=[['gpu']])]

    def _host_has_gpu(
            self
        ) -> bool:
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return True
            return False
        except FileNotFoundError:
            return False


    
    def _find_host_port(
            self
        ) -> None:
        host_port = self.DEFAULT_PORT
        while not is_port_free(host_port):
            print(f"port {host_port} not free. Checking next available port.")
            host_port += 1
        self.port_map[self.DEFAULT_PORT] = host_port

    def _set_volumes(
            self, 
            relative_host_paths: Optional[List[str]]=None,
            read_only=True,
        ) -> Dict[str, Dict[str, str]]:
        """
        Set volumes for mounting, relative to the path
        """
        if relative_host_paths is None:
            relative_host_paths = [] # "notebooks"

        mode = "ro" if read_only else "rw"

        self.mount_vols.update({
            os.path.join(self.script_dir, path): {
                "bind": f"/app/{path}",
                "mode": mode
            }
            for path in relative_host_paths
        })

    def _set_default_volumes(
            self
        ) -> None:
        """
        Set default volumes for pytorch-jupyter
        """
        self._set_volumes(
            relative_host_paths=["cuda_checker"],
            read_only=True
        )
        self._set_volumes(
            relative_host_paths=["notebooks"],
            read_only=False
        )

    def volumes(
            self, 
            search: str,
            subdir: str,
            exact: bool=False,
        ) -> None:
        """
        Search for paths on host to mount as volumes in your container.
        """
        search_dir = os.path.join(self.script_dir, subdir)
        matches = [
            os.path.join(subdir, path) 
            for path in os.listdir(search_dir) 
            if (exact and search==path)
            or (search.lower() in path.lower())
        ]
        self._set_volumes(matches, read_only=True)

    def _check_container_name(self):
        """
        If container name already exists, increment the name by 1
        """
        container_exists = lambda: self.client.containers.list(
            filters={
                "name": self.container_name
            }
        )
        containers = self.client.containers.list(all=True)
        pattern = re.compile(rf"^{self.container_name}(_([0-9]+))?$")
        desired_group_idx = 2
        matching_container_idxs = [
            int(pattern.match(container.name)[desired_group_idx] or 0) for container in containers
            if pattern.match(container.name)
        ] or [0]
        new_img_idx = max(matching_container_idxs) + 1
        self.container_name = self.container_name.rsplit("_", 1)[0] + "_" + str(new_img_idx)
        print("Using next available container name", self.container_name)
    
    def _open_browser(self, url):
        """
        Delayed browser to open jupyter, so docker is already running when it's up.
        """

        opener = lambda: [
                time.sleep(1.5), 
                webbrowser.open(url) if not self.error
                else None
            ]
        threading.Thread(target=opener).start()

    @property
    def _jupyter_url(self):
        return (
            f"http://127.0.0.1:{self.port_map.get(self.DEFAULT_PORT)}"
            f"/lab"
            f"?token={self.environment.get('JUPYTER_TOKEN', 'passwd')}"
        )

    def _run(self):
        """
        Open browser with jupyter notebook and start container.
        """
        self.launched=True
        self._check_container_name()
        self._find_host_port()
        self._open_browser(self._jupyter_url)
        try:
            self.client.containers.run(
                self.container_img + self.img_tag, 
                remove=True,
                device_requests=self._get_gpus(),
                name=self.container_name,
                ports=self.port_map,
                environment=self.environment,
                volumes=self.mount_vols, 
                detach=True
            )
            print(f"Detached container running in background at {self._jupyter_url}.")
            print(f"To stop and remove container, run \n  docker stop {self.container_name}")

        except Exception as e:
            self.error = True
            print("Error starting container,", str(e))
    
    def start(
            self, 
            image: str,
            search: str=None,
            subdir: str='models',
            exact: bool=False, 
            name: str='my-milvus-env'
        ) -> None:
        self.container_img = image
        if search is not None:
            self.volumes(search, subdir, exact)
        self.container_name = name

if __name__=="__main__":
    runner = Run()
    fire.Fire(runner)
    if not runner.launched:
        runner._run()