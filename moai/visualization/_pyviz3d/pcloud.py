from moai.utils.arguments import assert_numeric
from collections.abc import Callable
from pyviz3d.visualizer import Visualizer

import pyviz3d
import typing
import torch
import logging
import colour
import shutil
import numpy as np
import math
import os
import json

log = logging.getLogger(__name__)

__all__ = ['PointCloud']

class PointCloud(Callable):    
    def __init__(self,
        points:            typing.Union[str, typing.Sequence[str]],
        color:             typing.Union[str, typing.Sequence[str]],
        batch_percentage:   float=1.0,
        name:               str="default",
        port:               int=8098,
    ):
        self.name, self.port = name, port
        self.vertices = [points] if isinstance(points, str) else list(points)
        self.colors = [colour.Color(color)] if isinstance(color, str) else list(colour.Color(c) for c in color)
        self.batch_percentage = batch_percentage
        assert_numeric(log, 'batch percentage', batch_percentage, 0.0, 1.0)
        self.visualizer = Visualizer()
        # os.makedirs('pyviz3d', exist_ok=True)
        directory_source = os.path.realpath(
            os.path.join(os.path.dirname(pyviz3d.__file__), "src")
        )
        self.directory_destination = os.path.join(os.getcwd(), 'pyviz3d')
        shutil.copytree(directory_source, self.directory_destination)
        log.warning(f"PyViz3D visualization enabled. Run `python -m http.server {port}` @ {os.path.join(os.getcwd(), 'pyviz3d')} and then navigate to localhost:{port}.")

    def _save(self):
        # Assemble binary data files
        nodes_dict = {}
        for name, e in self.visualizer.elements.items():
            binary_file_path = os.path.join(self.directory_destination, name + ".bin")
            nodes_dict[name] = e.get_properties(name + ".bin")
            e.write_binary(binary_file_path)
        # Write json file containing all scene elements
        json_file = os.path.join(self.directory_destination, "nodes.json")
        with open(json_file, "w") as outfile:
            json.dump(nodes_dict, outfile)

    def __call__(self, tensors: typing.Dict[str, torch.Tensor]) -> None:
        for v, c in zip(self.vertices, self.colors):
            vertices = tensors[v].detach().cpu().numpy()
            b = math.ceil(self.batch_percentage * vertices.shape[0])
            for i in range(b):                
                colors = (np.array(c.rgb) * 255.0).astype(np.uint8)
                colors = colors[np.newaxis, ...].repeat(vertices.shape[1], 0)
                self.visualizer.add_points(f"{v}_{i}", vertices[i], colors, point_size=1.0)
        self._save()

'''
import http.server
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

PORT = 8080

Handler = http.server.SimpleHTTPRequestHandler

Handler.extensions_map={
        '.manifest': 'text/cache-manifest',
    '.html': 'text/html',
        '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.svg':	'image/svg+xml',
    '.css':	'text/css',
    '.js':	'text/javascript',
    '': 'application/octet-stream', # Default
    }

httpd = socketserver.TCPServer(("", PORT), Handler)

print("serving at port", PORT)
httpd.serve_forever()
'''