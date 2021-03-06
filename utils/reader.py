#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import xml.etree.ElementTree as ET
from copy import deepcopy

from utils.entity import Entity


class XMLReader(object):
    """docstring for XMLReader"""

    @staticmethod
    def read_scene(fname):
        def read_head(head):
            scene = {}
            for obj in head.iter('objectdef'):
                obj_name = obj.attrib['name']
                list_of_args = []
                for entity in obj:
                    entity_name = 'triangleset'
                    attrs = {k: list(map(float, v.split(','))) for k, v in entity.items() if k != 'name'}
                    if entity.tag == 'triangleset':
                        for triangle in entity:
                            if triangle.tag == 'triangle':
                                vs = [[p.get('x'), p.get('y'), p.get('z')] for p in triangle.iter('vertex')]
                            elif triangle.tag == 'trianglenext':
                                vs = [vs[1]] + [[p.get('x'), p.get('y'), p.get('z')] for p in triangle.iter('vertex')] + [vs[2]]
                            else:
                                raise AttributeError("Tag doesn't match either triangle or trianglenext.")

                            list_of_args.append({'vertices': vs, **attrs})
                    else:
                        entity_name = entity.tag
                        list_of_args.append(attrs)

                scene[obj_name] = Entity.create(entity_name, list_of_args, obj_name)

            return scene

        def read_body(body, scene):
            stack = [(body, -1)]
            idx = 0
            trans = {}

            obj_cnt = {obj_name: 0 for obj_name in scene.keys()}
            obj_trans = {}

            # perform DFS on XML's element tree
            while stack:
                node, pidx = stack.pop()

                for elem in reversed(node):
                    if elem.tag == 'object':
                        cidx = pidx
                        obj_name = elem.attrib['name']
                        suffix = obj_cnt[obj_name]
                        name = '{}:{}'.format(obj_name, suffix) if suffix else obj_name
                        if suffix:
                            scene[name] = deepcopy(scene[obj_name])
                        obj_cnt[obj_name] += 1
                        obj_trans[name] = []
                        while cidx != -1:
                            t, cidx = trans[cidx]
                            obj_trans[name].append((t.tag, {k: float(v) for k, v in t.items()}))

                    stack.append((elem, idx))
                    trans[idx] = (elem, pidx)
                    idx += 1

            for obj_name in obj_trans.keys():
                scene[obj_name].transform(obj_trans[obj_name][::-1])

            return scene

        # parse XML
        tree = ET.parse(fname)
        root = tree.getroot()

        # get objects definitions & objects translation+rotation info
        head = root.find('head')
        body = root.find('body')

        scene = read_head(head)
        scene = read_body(body, scene)

        return scene

    @staticmethod
    def read_tri(fname):
        # important info
        mat_c = []
        mat_e = []
        mat_p = []
        mat_spec = []
        mat_refl = []
        mat_refr = []

        # parse XML
        tree = ET.parse(fname)
        root = tree.getroot()

        # get object definitions
        obj_root = root.find('head').find('objectdef')

        # loop through objects
        for obj in obj_root:

            # loop through triangles
            for tri in obj:
                emission = tri.attrib['emission'].split(',')
                radiosity = tri.attrib['radiosity'].split(',')
                spec = tri.attrib['spec']
                refl = tri.attrib['refl']
                refr = tri.attrib['refr']
                mat_c.append([float(r) for r in radiosity])
                mat_e.append([float(e) for e in emission])
                mat_spec.append(float(spec))
                mat_refl.append(float(refl))
                mat_refr.append(float(refr))

                # loop through vertices
                vertices = []
                for vtx in tri:
                    vertices.append([float(vtx.attrib['x']),
                                     float(vtx.attrib['y']),
                                     float(vtx.attrib['z'])])
                mat_p.append(vertices)

        return np.array(mat_c, dtype=np.float32), \
            np.array(mat_p, dtype=np.float32), \
            np.array(mat_e, dtype=np.float32), \
            np.array(mat_spec, dtype=np.float32), \
            np.array(mat_refl, dtype=np.float32), \
            np.array(mat_refr, dtype=np.float32)
