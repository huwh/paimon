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
import org.apache.paimon.utils.IOUtils;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.Collections;

/** Utilities for loading persisted native vector training models. */
public class NativeVectorTrainingModels {

    private static final String IVF_PQ_NATIVE_INDEX_TYPE = "ivf_pq";

    private NativeVectorTrainingModels() {}

    public static VectorTrainingModel load(
            GlobalIndexFileReader fileReader, GlobalIndexIOMeta globalIndexFile)
            throws IOException {
        byte[] fileBytes;
        try (org.apache.paimon.fs.SeekableInputStream in = fileReader.getInputStream(globalIndexFile)) {
            fileBytes = IOUtils.readFully(in, false);
        }

        VectorGlobalIndexFileMeta.deserialize(fileBytes);
        byte[] payload = nativePayload(fileBytes);
        return NativeVectorGlobalModelTrainers.NativeVectorTrainingModel.load(
                IVF_PQ_NATIVE_INDEX_TYPE, Collections.emptyMap(), new ByteArrayInputStream(payload));
    }

    public static VectorTrainingModel load(byte[] fileBytes) throws IOException {
        VectorGlobalIndexFileMeta.deserialize(fileBytes);
        byte[] payload = nativePayload(fileBytes);
        return NativeVectorGlobalModelTrainers.NativeVectorTrainingModel.load(
                IVF_PQ_NATIVE_INDEX_TYPE, Collections.emptyMap(), new ByteArrayInputStream(payload));
    }

    private static byte[] nativePayload(byte[] fileBytes) throws IOException {
        try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(fileBytes))) {
            int metadataLength = in.readInt();
            if (metadataLength <= 0 || metadataLength > fileBytes.length - Integer.BYTES) {
                throw new IllegalArgumentException(
                        "Invalid vector global index metadata length: " + metadataLength);
            }
            int payloadOffset = Integer.BYTES + metadataLength;
            byte[] payload = new byte[fileBytes.length - payloadOffset];
            System.arraycopy(fileBytes, payloadOffset, payload, 0, payload.length);
            return payload;
        }
    }

}
