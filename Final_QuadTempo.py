import numpy as np
import cv2
import os
import timeit
import sys
import tensorflow as tf
from tqdm import tqdm
from pycuda import driver as drv
from pycuda import compiler, gpuarray, tools
import pycuda.autoinit

# load manta tools
sys.path.append("./tools")
import paramhelpers as ph
from GAN import GAN, lrelu
import fluiddataloader as FDL

# 직접 구현한 Quad Tree tools
from Quad_Tree_GPU import Qtree_GPU

class test_for_tempo(object):
    def __init__(self,
                 Set_data_dir="data\\256",
                 full_side_size=256,
                 start_frame=0,
                 total_frame=200,
                 data_type=np.float32):

        self.Set_data_dir = Set_data_dir
        self.simSizeLow = full_side_size
        self.start_frame = start_frame
        self.total_frame = total_frame
        self.depth = 4

        self.data_type = data_type

        self.gpu_num = 0

        self.train = False
        self.dropoutOutput = 1.0
        self.batch_norm = True

        self.upRes = 4  # fixed for now...
        self.simSizeHigh = self.simSizeLow * self.upRes
        self.bn = self.batch_norm
        self.rbId = 0
        self.overlap = 3
        self.set_parameter(512)

    # 폴더 생성 함수
    def make_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # tileSize에 맞게 Parameter 조정
    def set_parameter(self, tileSizeLow):
        self.tileSizeLow = tileSizeLow
        self.tileSizeHigh = self.tileSizeLow * self.upRes
        self.n_input = self.tileSizeLow ** 2
        self.n_output = self.tileSizeHigh ** 2
        self.n_inputChannels = 4
        self.n_input *= self.n_inputChannels

    # TempoGAN에서 제공하는 generator에 필요한 resBlock
    def resBlock(self, gan, inp, s1, s2, reuse, use_batch_norm, filter_size=3):
        global rbId

        filter = [filter_size, filter_size]
        filter1 = [1, 1]

        gc1, _ = gan.convolutional_layer(s1, filter, tf.nn.relu, stride=[1], name="g_cA%d" % rbId, in_layer=inp,
                                         reuse=reuse, batch_norm=use_batch_norm, train=self.train)  # ->16,64
        gc2, _ = gan.convolutional_layer(s2, filter, None, stride=[1], name="g_cB%d" % rbId, reuse=reuse,
                                         batch_norm=use_batch_norm, train=self.train)  # ->8,128
        # shortcut connection
        gs1, _ = gan.convolutional_layer(s2, filter1, None, stride=[1], name="g_s%d" % rbId, in_layer=inp, reuse=reuse,
                                         batch_norm=use_batch_norm, train=self.train)  # ->16,64
        resUnit1 = tf.nn.relu(tf.add(gc2, gs1))
        rbId += 1
        return resUnit1

    # TempoGAN에서 제공하는 generator Layer
    def gen_resnet(self, _in, num, reuse=False, use_batch_norm=False):
        global rbId
        with tf.variable_scope("generator_" + str(num), reuse=reuse) as scope:

            _in = tf.reshape(_in, shape=[-1, self.tileSizeLow, self.tileSizeLow, self.n_inputChannels])  # NHWC

            rbId = 0
            gan = GAN(_in)

            gan.max_depool()
            inp = gan.max_depool()
            ru1 = self.resBlock(gan, inp, self.n_inputChannels * 2, self.n_inputChannels * 8, reuse, use_batch_norm, 5)

            ru2 = self.resBlock(gan, ru1, 128, 128, reuse, use_batch_norm, 5)
            inRu3 = ru2
            ru3 = self.resBlock(gan, inRu3, 32, 8, reuse, use_batch_norm, 5)
            ru4 = self.resBlock(gan, ru3, 2, 1, reuse, False, 5)
            resF = tf.reshape(ru4, shape=[-1, self.n_output])
            return resF

    # TempoGAN Super-resolution 함수
    def tempoGAN(self, data__, key__):
        resultTiles = []
        sampler = self.sampler_0
        if self.tileSizeLow == self.simSizeLow:
            sampler = self.sampler_0
        elif self.tileSizeLow == self.simSizeLow // 2 + self.overlap * 2:
            sampler = self.sampler_1
        elif self.tileSizeLow == self.simSizeLow // 4 + self.overlap * 2:
            sampler = self.sampler_2
        elif self.tileSizeLow == self.simSizeLow // 8 + self.overlap * 2:
            sampler = self.sampler_3
        elif self.tileSizeLow == self.simSizeLow // 16 + self.overlap * 2:
            sampler = self.sampler_4

        # Super_resolution 진행
        for tileno in tqdm(range(data__.shape[0])):
            batch_xs_in = np.reshape(data__[tileno], [-1, self.n_input])
            results = self.sess.run(sampler, feed_dict={self.x: batch_xs_in, self.keep_prob: 1.0, self.train: False})
            results = np.array(results)
            results_reshape = np.reshape(results, [self.tileSizeHigh, self.tileSizeHigh, 1])

            resultTiles.append(results_reshape)

        # Super_resolution 된 데이터 하나로 합치기
        if key__[0][0] == 0:
            result_hd = resultTiles[0]
        else:
            result_hd = self.QT.data_sum_overlap(key__, resultTiles, self.simSizeHigh, self.simSizeHigh)

        return result_hd

    # 패치 별 최댓값 구하는 함수
    # GPU Reduction을 활용하여 최댓값 구하기
    def Gen_Maxmatrix(self, patch_size):
        # Pycuda 코드
        func = """
                   #include <stdio.h>
                   __global__ void reduction_max(float *input, int input_N, float *output)
                   {
                       unsigned int size = input_N;
                       __shared__ float Max_x[4][4];
                       unsigned int output_size = size * 0.5;

                       unsigned int ty = threadIdx.y + blockDim.y * blockIdx.y;
                       unsigned int tx = threadIdx.x + blockDim.x * blockIdx.x;

                       unsigned int tid_x = threadIdx.x;
                       unsigned int tid_y = threadIdx.y;
                       Max_x[tid_x][tid_y] = input[tx + (ty * size)];
                       __syncthreads();

                       for(unsigned int s=1; s < blockDim.x; s*=2)
                       {   
                           int index = 2 * s * tid_x;
                           if(index < blockDim.x)
                           {
                               Max_x[tid_x][tid_y] = (Max_x[tid_x + s][tid_y] > Max_x[tid_x][tid_y]) ? Max_x[tid_x + s][tid_y] : Max_x[tid_x][tid_y];
                           }
                       }

                       if(tid_x == 0)
                       {
                           input[tx + (ty * size)] = Max_x[tid_x][tid_y];
                       }

                       Max_x[tid_x][tid_y] = input[tx + (ty * size)];
                       __syncthreads();        

                       for(unsigned int r=1; r < blockDim.y; r*=2)
                       {      
                           int index = 2 * r * tid_x;
                           if(index < blockDim.y)
                           {
                               Max_x[tid_x][tid_y] = (Max_x[tid_x][tid_y + r] > Max_x[tid_x][tid_y]) ? Max_x[tid_x][tid_y + r] : Max_x[tid_x][tid_y];
                           }
                       }

                       if(tid_x == 0)
                       {
                           if(tid_y == 0)
                           {
                               output[blockIdx.x + blockIdx.y * output_size] = Max_x[tid_x][tid_y];
                               __syncthreads();
                           }
                       }
                   }
                   """

        if (os.system("cl.exe")):
            os.environ[
                'PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\HostX64\x64"
        if (os.system("cl.exe")):
            raise RuntimeError("cl.exe still not found, path probably incorrect")

        mod = compiler.SourceModule(func)
        reduction_max = mod.get_function("reduction_max")

        data_shape_size = self.simSizeLow // patch_size

        # 원하는 패치 개수만큼 최댓값 굳하는 함수
        def Max(input_data, input_N):
            input = input_data
            block_N = 2
            grid_N = int(input_N / block_N)
            output = np.zeros((input_N // 2, input_N // 2), np.float32)
            reduction_max(drv.In(input), np.int32(input_N), drv.Out(output), block=(block_N, block_N, 1),
                          grid=(grid_N, grid_N))
            if (output.shape[0] == data_shape_size):
                return output
            input = output
            input_N = input.shape[0]
            result = Max(input, input_N)

            return result

        Max_Matrix_list = []

        self.input_data = self.data_setting()

        # 원하는 프레임만큼 돌면서 Max_Martix를 만들어서 리스트로 저장
        data_count = 0
        for _ in tqdm(range(self.start_frame, self.total_frame)):
            data = self.input_data[data_count, :, :, :, 0]
            data = np.reshape(data, (self.simSizeLow, self.simSizeLow, 1))
            data = np.ascontiguousarray(data, dtype=np.float32)
            Max_Matrix = Max(data, self.simSizeLow)
            Max_Matrix_list.append(Max_Matrix)
            data_count += 1
        return Max_Matrix_list

    # TempoGAN에서 제공하는 MantaFlow 데이터 읽어오는 함수
    def data_setting(self):
        loadPath = "data/" + str(self.simSizeLow) + "/high/"
        channelLayout_low = 'd'
        lowfilename = "density_%04d.uni"
        mfl = ["density"]
        channelLayout_low += ',vx,vy,vz'
        mfl = np.append(mfl, "vel")
        floader = FDL.FluidDataLoader(print_info=1, base_path=loadPath, filename=lowfilename, oldNamingScheme=False,
                                      filename_y=None, filename_index_min=self.start_frame, filename_index_max=self.total_frame,
                                      data_fraction=1.0, multi_file_list=mfl, multi_file_list_y=None)
        inputx, _, _ = floader.get()
        return inputx

    # Quad Tree를 활용한 Super-resolution 함수
    def Run_QuadTreeSR(self, patch_size):
        self.patch_size = patch_size
        self.save_dir = "result/Octempo_" + str(self.simSizeLow) + "_" + str(patch_size)
        self.make_folder(self.save_dir)
        f_time = open("result/time_check/Quadtempo_" + str(self.simSizeLow) + "_" + str(patch_size) + ".txt", 'w')

        pre_process_time_start = timeit.default_timer()

        ph.checkUnusedParams()

        modelPath = "model/model_0199_final.ckpt"

        # 각 패치 별 최댓값으로 만든 메트릭스 생성
        Max_Matrix_list = self.Gen_Maxmatrix(patch_size)
        Max_Matrix_list_size = 0

        self.x = tf.placeholder(tf.float32, shape=[None, None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.train = tf.placeholder(tf.bool)

        # 각 패치 크기 별로 Layer 설정
        self.set_parameter(self.simSizeLow)
        self.sampler_0 = self.gen_resnet(self.x, 0, reuse=False, use_batch_norm=True)
        self.set_parameter(self.simSizeLow // 2 + self.overlap * 2)
        self.sampler_1 = self.gen_resnet(self.x, 1, reuse=False, use_batch_norm=True)
        self.set_parameter(self.simSizeLow // 4 + self.overlap * 2)
        self.sampler_2 = self.gen_resnet(self.x, 2, reuse=False, use_batch_norm=True)
        self.set_parameter(self.simSizeLow // 8 + self.overlap * 2)
        self.sampler_3 = self.gen_resnet(self.x, 3, reuse=False, use_batch_norm=True)
        self.set_parameter(self.simSizeLow // 16 + self.overlap * 2)
        self.sampler_4 = self.gen_resnet(self.x, 4, reuse=False, use_batch_norm=True)

        # TempoGAN 모델 불러오기
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        saver = tf.train.Saver()
        saver.restore(self.sess, modelPath)

        pre_process_time = timeit.default_timer() - pre_process_time_start

        f_time.write("0" + " " + str(pre_process_time))
        for frame in range(self.start_frame, self.total_frame):
            print("frame = ", frame)
            one_frame_start_time = timeit.default_timer()

            data = self.input_data[Max_Matrix_list_size]
            data = np.reshape(data, (self.simSizeLow, self.simSizeLow, 4))
            data = np.ascontiguousarray(data, dtype=np.float32)

            self.QT = Qtree_GPU(data, Max_Matrix_list[Max_Matrix_list_size], patch_size)

            # 패치 크기 및 키 별로 받기
            key0, data0, key1, data1, key2, data2, key3, data3, key4, data4, key_t, data_t = self.QT.set_data_quadtree()
            data0 = np.array(data0)
            data1 = np.array(data1)
            data2 = np.array(data2)
            data3 = np.array(data3)
            data4 = np.array(data4)
            data_t = np.array(data_t)

            # 결과로 저장할 배열 설정
            final_result = np.zeros([self.simSizeLow * 4, self.simSizeLow * 4, 1]).astype(np.float32)

            # 패치 크기마다 데이터가 있을 때 Super-resolution
            # final result에 각각 더해주기
            if data0.shape[0] > 0:
                self.set_parameter(data0.shape[1])
                result0 = self.tempoGAN(data0, key0)
                final_result += result0
            if data1.shape[0] > 0:
                self.set_parameter(data1.shape[1])
                result1 = self.tempoGAN(data1, key1)
                final_result += result1
            if data2.shape[0] > 0:
                self.set_parameter(data2.shape[1])
                result2 = self.tempoGAN(data2, key2)
                final_result += result2
            if data3.shape[0] > 0:
                self.set_parameter(data3.shape[1])
                result3 = self.tempoGAN(data3, key3)
                final_result += result3
            if data4.shape[0] > 0:
                self.set_parameter(data4.shape[1])
                result4 = self.tempoGAN(data4, key4)
                final_result += result4
            if data_t.shape[0] > 0:
                self.set_parameter(data_t.shape[1])
                result_t = self.tempoGAN(data_t, key_t)
                final_result += result_t

            # 결과 이미지로 저장
            final_result = np.uint8(np.clip(final_result * 255, 0, 255))
            final_result = cv2.flip(final_result, 0)
            cv2.imwrite(self.save_dir + "/%05d" % (frame + 1) + ".jpg", final_result)
            one_frame_time = timeit.default_timer() - one_frame_start_time
            f_time.write("\n" + str(frame+1) + " " + str(one_frame_time))
            Max_Matrix_list_size += 1

        f_time.close()
        self.sess.close()
