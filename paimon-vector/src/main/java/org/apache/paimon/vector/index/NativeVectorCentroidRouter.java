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

import org.apache.paimon.globalindex.GlobalIndexIOMeta;
import org.apache.paimon.globalindex.io.GlobalIndexFileReader;
import org.apache.paimon.predicate.VectorSearch;

import java.io.Closeable;
import java.io.IOException;

/** Routes IVF queries to centroid shard ids using a persisted global training model. */
public class NativeVectorCentroidRouter implements Closeable {

    private static final String NPROBE_PARAMETER = "ivf.nprobe";

    private final VectorTrainingModel model;

    public NativeVectorCentroidRouter(
            GlobalIndexFileReader fileReader, GlobalIndexIOMeta globalIndexFile)
            throws IOException {
        this.model = NativeVectorTrainingModels.load(fileReader, globalIndexFile);
    }

    public int[] route(float[] queryVector, int nprobe) {
        return model.centroids().nearestCentroids(queryVector, nprobe);
    }

    public int[] route(VectorSearch vectorSearch) {
        return route(vectorSearch.vector(), nprobe(vectorSearch));
    }

    public VectorCentroidModel centroidModel() {
        return model.centroids();
    }

    private static int nprobe(VectorSearch vectorSearch) {
        String value = vectorSearch.options().get(NPROBE_PARAMETER);
        if (value == null) {
            return vectorSearch.limit();
        }
        try {
            int nprobe = Integer.parseInt(value);
            if (nprobe <= 0) {
                throw new IllegalArgumentException(
                        "Invalid value for '" + NPROBE_PARAMETER + "': " + value);
            }
            return nprobe;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid value for '" + NPROBE_PARAMETER + "': " + value, e);
        }
    }

    @Override
    public void close() throws IOException {
        model.close();
    }
}
