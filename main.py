import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
import pprint

from Final_QuadTempo import test_for_tempo

pp = pprint.PrettyPrinter()

flags = tf.app.flags

#기본 Parameter Setting
flags.DEFINE_string("Set_data_dir", "data/512/high/", "Set Training Data path")
flags.DEFINE_integer("full_side_size", 512, "data one side size")
flags.DEFINE_integer("start_frame", 0, "test_data start frame")
flags.DEFINE_integer("total_frame", 200, "test_data total frame")
flags.DEFINE_string("scene", "", "name of scene")

FLAGS = flags.FLAGS

def main(_):

    pp.pprint(flags.FLAGS.__flags)

    import timeit
    start_time = timeit.default_timer()
    Set = test_for_tempo(
        Set_data_dir=FLAGS.Set_data_dir,
        full_side_size=FLAGS.full_side_size,
        start_frame=FLAGS.start_frame,
        total_frame=FLAGS.total_frame,
    )
    print("Set_GPU_Octree_training_data_start")
    Set.Run_QuadTreeSR(64)
    end_time = timeit.default_timer() - start_time
    print("total time is = ", end_time)

if __name__ == '__main__':
    tf.compat.v1.app.run()
