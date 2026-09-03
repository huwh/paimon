/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.paimon.vector.index;

/** Options and constants for the native vector index. */
public class NativeVectorIndexOptions {

    public static final String IVF_PQ_INDEX_TYPE = "ivf-pq";

    public static final String IVF_PQ_SHARD_OPTION = "ivf.pq.shard";

    public static final String CENTROID_TRAIN_MODE_OPTION = "ivf.pq.train.mode";

    public static final String CENTROID_TRAIN_MODE_LOCAL = "local";

    public static final String CENTROID_TRAIN_MODE_DISTRIBUTED = "distributed";

    public static final String DISTRIBUTED_TRAIN_MAX_ITERATIONS_OPTION =
            "ivf.pq.train.distributed.max-iterations";

    public static final String DISTRIBUTED_TRAIN_TOLERANCE_OPTION =
            "ivf.pq.train.distributed.tolerance";

    public static final String DISTRIBUTED_TRAIN_PARTITION_BATCH_SIZE_OPTION =
            "ivf.pq.train.distributed.partition-batch-size";

    public static final String DISTRIBUTED_TRAIN_MAX_STATE_BYTES_OPTION =
            "ivf.pq.train.distributed.max-state-bytes";

    public static final String DISTRIBUTED_TRAIN_MAX_PARTIAL_BYTES_OPTION =
            "ivf.pq.train.distributed.max-partial-bytes";

    public static final String DISTRIBUTED_TRAIN_MAX_DRIVER_PARTIAL_BYTES_OPTION =
            "ivf.pq.train.distributed.max-driver-partial-bytes";

    public static final String DISTRIBUTED_TRAIN_MAX_BOOTSTRAP_BYTES_OPTION =
            "ivf.pq.train.distributed.max-bootstrap-bytes";

    public static final String DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES_OPTION =
            "ivf.pq.train.distributed.max-pq-sample-bytes";

    public static final long DEFAULT_DISTRIBUTED_TRAIN_MIN_PQ_SAMPLE_ROWS = 65_536L;

    public static final long DEFAULT_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS_PER_CENTROID = 64L;

    public static final long MIN_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS = 256L;

    /** Native PQ K-means uses at most 256 samples per codeword (256 * 256). */
    public static final long MAX_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS = 65_536L;

    public static final long DEFAULT_DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES = 256L * 1024 * 1024;

    /** The native sparse accumulator format rejects a larger serialized partition partial. */
    public static final int MAX_DISTRIBUTED_TRAIN_PARTIAL_BYTES = 64 * 1024 * 1024;

    public static final String CENTROID_BACKEND_OPTION = "ivf.pq.centroid.backend";

    public static final String CENTROID_BACKEND_NATIVE = "native";

    public static final String CENTROID_ASSIGN_LAZY_STREAMING_OPTION =
            "ivf.pq.centroid.assign.lazy-streaming";

    public static final String TRAIN_SAMPLE_ROWS_OPTION = "train.sample-rows";

    public static final String VECTOR_ROUTING_MODEL_FILE_PREFIX =
            "vector-ivf-pq-global-index-training-model";

    public static final String GLOBAL_INDEX_FILE_EXTENSION = ".index";

    private NativeVectorIndexOptions() {}
}
