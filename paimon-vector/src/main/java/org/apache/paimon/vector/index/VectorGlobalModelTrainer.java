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

import javax.annotation.Nullable;

import java.io.Closeable;
import java.io.IOException;

/**
 * Trainer for centroid-sharded IVF/PQ global models.
 *
 * <p>The input side intentionally mirrors {@code GlobalIndexSingleColumnWriter#write(Object,
 * long)}, so the build side can reuse the existing single-column global-index row reading flow. It
 * does not extend {@code GlobalIndexSingleColumnWriter} because the normal writer {@code finish()}
 * returns index file entries, while model training returns a reusable {@link VectorTrainingModel}
 * for later model persistence, centroid assignment and shard writer creation.
 */
public interface VectorGlobalModelTrainer extends Closeable {

    /**
     * Writes one sampled training vector into the trainer.
     *
     * <p>This method intentionally mirrors {@code GlobalIndexSingleColumnWriter#write(Object,
     * long)}. The row id is absolute because centroid-sharded files use absolute row ids.
     */
    void write(@Nullable Object vector, long absoluteRowId) throws IOException;

    /** Completes local training and returns the resulting reusable training model. */
    VectorTrainingModel finishTraining() throws IOException;
}
