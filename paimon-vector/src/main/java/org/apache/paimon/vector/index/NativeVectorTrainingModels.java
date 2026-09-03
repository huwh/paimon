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

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;

/** Utilities for loading persisted native vector training models. */
public class NativeVectorTrainingModels {

    private static final String IVF_PQ_NATIVE_INDEX_TYPE = "ivf_pq";

    private NativeVectorTrainingModels() {}

    public static VectorTrainingModel load(
            GlobalIndexFileReader fileReader, GlobalIndexIOMeta globalIndexFile)
            throws IOException {
        try (org.apache.paimon.fs.SeekableInputStream in =
                fileReader.getInputStream(globalIndexFile)) {
            return loadArtifact(in, globalIndexFile.fileSize());
        }
    }

    public static VectorTrainingModel load(byte[] fileBytes) throws IOException {
        try (ByteArrayInputStream in = new ByteArrayInputStream(fileBytes)) {
            return loadArtifact(in, fileBytes.length);
        }
    }

    private static VectorTrainingModel loadArtifact(InputStream in, long artifactSize)
            throws IOException {
        VectorGlobalIndexFileMeta.ArtifactHeader header =
                VectorGlobalIndexFileMeta.readArtifactHeader(in, artifactSize);
        byte[] payload = readPayload(in, header.payloadSize());
        String actualDigest = VectorModelDigest.sha256(payload);
        if (header.metadata().hasModelIdentity()
                && !header.metadata().modelDigest().equals(actualDigest)) {
            throw new IOException(
                    "Vector training model digest mismatch: expected "
                            + header.metadata().modelDigest()
                            + ", but was "
                            + actualDigest);
        }
        VectorTrainingModel model =
                NativeVectorGlobalModelTrainers.NativeVectorTrainingModel.load(
                        IVF_PQ_NATIVE_INDEX_TYPE, Collections.emptyMap(), payload);
        int actualNlist = model.centroids().nlist();
        if (header.metadata().hasModelIdentity() && actualNlist != header.metadata().nlist()) {
            try {
                model.close();
            } catch (IOException closeFailure) {
                // The metadata mismatch remains the actionable failure.
            }
            throw new IOException(
                    "Vector training model nlist mismatch: expected "
                            + header.metadata().nlist()
                            + ", but was "
                            + actualNlist);
        }
        return model;
    }

    private static byte[] readPayload(InputStream in, long payloadSize) throws IOException {
        if (payloadSize <= 0 || payloadSize > Integer.MAX_VALUE) {
            throw new IOException("Unsupported vector training model payload size: " + payloadSize);
        }
        byte[] payload = new byte[(int) payloadSize];
        DataInputStream dataInput =
                in instanceof DataInputStream ? (DataInputStream) in : new DataInputStream(in);
        dataInput.readFully(payload);
        if (dataInput.read() != -1) {
            throw new IOException("Vector training model artifact contains trailing bytes.");
        }
        return payload;
    }
}
