import cv2
import numpy as np
import os
import timeit
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Quad Tree를 만들기 위한 Node Class 설정
class Node(object):
    def __init__(self, data):
        # 각 노드가 가지는 속성 데이터 키 상태값
        self.key = 0
        self.data = data
        self.State = 0

        self.father_node = None
        self.Quad_1 = self.Quad_2 = self.Quad_3 = self.Quad_4 = None


class Qtree_GPU_(object):
    def __init__(self, data, Max_Matrix, patch_size):
        self.root = None

        self.root_data = data
        self.node_list = []
        self.present_node_list = []

        self.patchsize = patch_size
        self.threshold_min = 0.01

        # Overlap = 데이터를 겹쳐서 Super-resolution 한 뒤에 겹친부분 삭제하면 Artifact가 사라지는 효과 있음
        # TempoGAN 논문 확인
        self.overlap_size = 3

        self.data_type = np.float32
        self.width = len(data[0])
        self.height = len(data)

        # Empty = Super-resolution 을 할 필요 없는 공간
        # S_SRR = Super-resolution 을 해야 할 공간
        # MIX = SRR과 Empty가 섞여 있는 공간
        self.S_Empty = "Empty"
        self.S_SRR = "S_SRR"
        self.S_Mix = "Mix"

        self.max_patch_size = 512           # Quad Tree 최상위 노드의 데이터 크기 제한(GPU 메모리 문제)
        self.Max_Matrix = Max_Matrix

        # 최하단 노드의 Patch 개수
        self.terminal_patch_num_x = self.width // self.patchsize
        self.terminal_patch_num_y = self.height // self.patchsize
        self.terminal_depth = 0

        self.present_patch_num_x = self.terminal_patch_num_x
        self.present_patch_num_y = self.terminal_patch_num_y
        self.present_depth = 0

        self.depth0_SRR_key = []
        self.depth1_SRR_key = []
        self.depth2_SRR_key = []
        self.depth3_SRR_key = []
        self.depth4_SRR_key = []
        self.terminal_SRR_key = []

        self.depth0_SRR_data = []
        self.depth1_SRR_data = []
        self.depth2_SRR_data = []
        self.depth3_SRR_data = []
        self.depth4_SRR_data = []
        self.terminal_SRR_data = []

        self.run(data)

    # 최하단 노드의 깊이 값 계산
    def terminal_depth_compute(self, width, patchsize):
        tmp = width // patchsize
        i = 0
        while True:
            if tmp == pow(2, i):
                self.terminal_depth = i
                break
            i = i + 1

    def terminal_state_check(self, y, x):
        biggest_data_value = self.Max_Matrix[y, x]
        # Patch에 해당하는 최대값이 threshold 보다 작으면 비어있는 공간(Super-resolution이 필요 없는 공간으로 상태 설정)
        # threshold 보다 크면 Super-resoluiton을 해야하는 공간으로 상태 설정
        if self.threshold_min > biggest_data_value:
            State = self.S_Empty
        else:
            State = self.S_SRR
        return State

    # Quad Tree를 만들기 위해 최하단 노드 생성
    def Set_terminal_node(self, data):
        self.terminal_depth_compute(self.width, self.patchsize) # 깊이 계산

        # 모든 패치를 돌아가며 노드 생성 및 설정
        for i in range(self.terminal_patch_num_y):
            node_list_x = []
            for j in range(self.terminal_patch_num_x):

                # 최하단 노드에 들어갈 데이터 Slice
                terminal_data = data[i * self.patchsize: (i + 1) * self.patchsize, j * self.patchsize: (j + 1) * self.patchsize, :]
                # 최하단 노드에 데이터 키 상태 넣어주기
                node = Node(terminal_data)
                key = (self.terminal_depth, j+1, i+1)
                State = self.terminal_state_check(i, j)
                node.key = key
                node.State = State
                node_list_x.append(node)
            # 노드 리스트에 추가
            self.node_list.append(node_list_x)
        self.present_depth = self.terminal_depth
        self.present_node_list = self.node_list



    # Quad Tree를 만들 때, 자식 노드 4개의 상태값을 보고 부모 노드 상태값 결정 및 데이터 병합
    def Neighbor_State_Search(self, Grid_x, Grid_y):

        # 자식 노드 4개 상태 확인
        xystate = self.present_node_list[Grid_y - 1][Grid_x - 1].State
        x_ystate = self.present_node_list[Grid_y - 2][Grid_x - 1].State
        xy_state = self.present_node_list[Grid_y - 1][Grid_x - 2].State
        x_y_state = self.present_node_list[Grid_y - 2][Grid_x - 2].State

        # 자식 노드 4개 데이터 병합
        xystate_data = self.present_node_list[Grid_y - 1][Grid_x - 1].data
        x_ystate_data = self.present_node_list[Grid_y - 2][Grid_x - 1].data
        xy_state_data = self.present_node_list[Grid_y - 1][Grid_x - 2].data
        x_y_state_data = self.present_node_list[Grid_y - 2][Grid_x - 2].data
        img_line_1 = np.concatenate((x_y_state_data, x_ystate_data), axis=1)
        img_line_2 = np.concatenate((xy_state_data, xystate_data), axis=1)
        full_data = np.concatenate((img_line_1, img_line_2), axis=0)

        # 자식 노드 상태 확인하여 부모노드 상태값 및 데이터 리턴
        if xystate == x_ystate == xy_state == x_y_state == self.S_SRR:
            return self.S_SRR, full_data
        elif xystate == x_ystate == xy_state == x_y_state == self.S_Empty:
            return self.S_Empty, full_data
        else:
            return self.S_Mix, full_data

    # 부모 노드와 자식 노드 연결
    def Connect_Quadtree(self, node, Grid_x, Grid_y):
        node.Quad_1 = self.present_node_list[Grid_y - 2][Grid_x - 2]
        node.Quad_2 = self.present_node_list[Grid_y - 2][Grid_x - 1]
        node.Quad_3 = self.present_node_list[Grid_y - 1][Grid_x - 2]
        node.Quad_4 = self.present_node_list[Grid_y - 1][Grid_x - 1]

        return node

    # 트리 생성 함수
    def Set_tree(self, node, data):

        # 부모 노드가 최상위 노드일 때
        if self.present_depth == 1:
            Grid_x = self.present_patch_num_x
            Grid_y = self.present_patch_num_y
            self.present_depth = self.present_depth - 1
            for y in range(2, Grid_y + 1, 2):
                for x in range(2, Grid_x + 1, 2):
                    # 부모 노드의 데이터 키 상태값 계산해서 부모노드 생성 및 설정
                    root_State, root_data = self.Neighbor_State_Search(x, y)
                    node = Node(root_data)
                    self.present_node_list[y - 1][x - 1].father_node = node
                    key = (self.present_depth, x - 1, y - 1)

                    node.key = key
                    node.State = root_State
                    self.Connect_Quadtree(node, x, y)
            self.root = node
        # 부모 노드가 최상위 노드가 아니면
        else:
            Grid_x = self.present_patch_num_x
            Grid_y = self.present_patch_num_y
            self.present_depth = self.present_depth - 1
            self.present_patch_num_x = self.present_patch_num_x // 2
            self.present_patch_num_y = self.present_patch_num_y // 2
            node_list = []
            for y in range(2, Grid_y + 1, 2):
                node_list_x = []
                for x in range(2, Grid_x + 1, 2):
                    # 부모 노드의 데이터 키 상태값 계산해서 부모노드 생성 및 설정
                    father_State, father_data = self.Neighbor_State_Search(x, y)
                    node = Node(father_data)
                    self.present_node_list[y - 1][x - 1].father_node = node
                    key = (self.present_depth, x // 2, y // 2)

                    node.key = key
                    node.State = father_State
                    node_list_x.append(node)
                node_list.append(node_list_x)

            for y in range(2, Grid_y + 1, 2):
                for x in range(2, Grid_x + 1, 2):
                    node = self.present_node_list[y - 1][x - 1].father_node
                    self.Connect_Quadtree(node, x, y)
            self.present_node_list = node_list

            # 재귀 함수로 돌면서 Quad Tree 생성
            self.Set_tree(self.root, data)

        return node

    # Overlap 영역까지 Data 분할하여 Super-resolution 해야 할 데이터 리턴
    def node_data_setting_overlap(self, key):
        overlap_size = self.overlap_size

        side_size = self.width // pow(2, key[0])
        # 키의 0번째가 0이면 깊이가 0이라는 뜻 이므로, 깊이가 0이면 Root Node 이다.
        # Root Node Super-resolution은 overlap 불필요
        if key[0] == 0:
            data = self.root_data
        else:
            x = key[1]
            y = key[2]

            x_start = (x - 1) * side_size - overlap_size
            y_start = (y - 1) * side_size - overlap_size

            x_end = x * side_size + overlap_size
            y_end = y * side_size + overlap_size

            if x == 1:
                x_start = 0
                x_end = x * side_size + overlap_size * 2
            if y == 1:
                y_start = 0
                y_end = y * side_size + overlap_size * 2

            if x * side_size + overlap_size > self.width:
                x_start = (x - 1) * side_size - overlap_size * 2
                x_end = self.width
            if y * side_size + overlap_size > self.width:
                y_start = (y - 1) * side_size - overlap_size * 2
                y_end = self.width

            data = self.root_data[y_start: y_end, x_start: x_end, :]
        return data

    # 나눠진 크기 별로 데이터 및 키 값 리턴
    # TempoGAN 모델에 맞추기 위한 단계
    def set_data_quadtree(self):
        key0, data0, key1, data1, key2, data2, key3, data3, key4, data4, key_t, data_t = self._set_data_quadtree(self.root)
        return key0, data0, key1, data1, key2, data2, key3, data3, key4, data4, key_t, data_t

    # Tree 순회하며 크기 별로 데이터 모으기
    def _set_data_quadtree(self, node):
        depth_tmp = node.key[0]
        depth_pow = pow(2, depth_tmp)
        node_side_size = self.width // depth_pow
        if node.key[0] != self.terminal_depth:
            if node.State == node.Quad_1.State == node.Quad_2.State == node.Quad_3.State == node.Quad_4.State == self.S_SRR \
                    and node_side_size <= self.max_patch_size:
                ''''''
                if node.key[0] == 0:
                    self.depth0_SRR_key.append(node.key)
                    self.depth0_SRR_data.append(self.root_data)
                elif node.key[0] == 1:
                    self.depth1_SRR_key.append(node.key)
                    self.depth1_SRR_data.append(self.node_data_setting_overlap(node.key))
                elif node.key[0] == 2:
                    self.depth2_SRR_key.append(node.key)
                    self.depth2_SRR_data.append(self.node_data_setting_overlap(node.key))
                elif node.key[0] == 3:
                    self.depth3_SRR_key.append(node.key)
                    self.depth3_SRR_data.append(self.node_data_setting_overlap(node.key))
                elif node.key[0] == 4:
                    self.depth4_SRR_key.append(node.key)
                    self.depth4_SRR_data.append(self.node_data_setting_overlap(node.key))

            elif node.State == self.S_Empty:
                ''''''
            else:
                self._set_data_quadtree(node.Quad_1)
                self._set_data_quadtree(node.Quad_2)
                self._set_data_quadtree(node.Quad_3)
                self._set_data_quadtree(node.Quad_4)
        else:
            if node.State == self.S_Empty:
                ''''''
            else:
                self.terminal_SRR_key.append(node.key)
                self.terminal_SRR_data.append(self.node_data_setting_overlap(node.key))

        return self.depth0_SRR_key, self.depth0_SRR_data, self.depth1_SRR_key, self.depth1_SRR_data,\
               self.depth2_SRR_key, self.depth2_SRR_data, self.depth3_SRR_key, self.depth3_SRR_data,\
               self.depth4_SRR_key, self.depth4_SRR_data, self.terminal_SRR_key, self.terminal_SRR_data


    # Overlap 부분 제거하면서 Data 합치기
    def data_sum_overlap(self, key, data, width, height):
        overlap_size = self.overlap_size
        base = np.zeros((width, height, 1), self.data_type)
        array_size = len(key)
        for i in range(array_size):
            side_size = width // pow(2, key[i][0])
            x = key[i][1]
            y = key[i][2]
            data_ = data[i]

            base_x_start = (x - 1) * side_size
            base_y_start = (y - 1) * side_size

            base_x_end = x * side_size
            base_y_end = y * side_size

            data_x_start = overlap_size * 4
            data_y_start = overlap_size * 4

            data_x_end = overlap_size * 4 + side_size
            data_y_end = overlap_size * 4 + side_size

            # 각각 데이터가 Boundary에 있을 때 따로 설정
            if x == 1:
                base_x_start = 0
                base_x_end = side_size
                data_x_start = 0
                data_x_end = side_size
            if y == 1:
                base_y_start = 0
                base_y_end = side_size
                data_y_start = 0
                data_y_end = side_size

            if x * side_size + overlap_size > self.width * 4:
                base_x_start = (x - 1) * side_size
                base_x_end = self.width * 4
                data_x_start = overlap_size * 4 * 2
                data_x_end = side_size + overlap_size * 4 * 2
            if y * side_size + overlap_size > self.width * 4:
                base_y_start = (y - 1) * side_size
                base_y_end = self.width * 4
                data_y_start = overlap_size * 4 * 2
                data_y_end = side_size + overlap_size * 4 * 2

            base[base_y_start: base_y_end, base_x_start: base_x_end, :]\
                = data_[data_y_start: data_y_end, data_x_start: data_x_end, :]

        return base

    def run(self, data):
        self.Set_terminal_node(data)
        self.Set_tree(self.root, data)



