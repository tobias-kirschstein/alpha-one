"""Simple example on how to log scalars and images to tensorboard without tensor ops.
License: BSD License 2.0
"""
__author__ = "Michael Gygli"

import tensorflow as tf
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np


class TensorboardLogger(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()

    def log_scalar(self, name, value, step=None):
        tf.summary.scalar(name, value, step)

    def log_images(self, name, images, step=None):
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            tf.summary.image(name, s.getvalue(), step, description='%s/%d' % (name, nr))

    def log_histogram(self, name, values, step, bins=1000):
        # TODO: revise
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()