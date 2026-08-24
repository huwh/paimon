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

import org.apache.paimon.globalindex.GlobalIndexSingleColumnWriter;
import org.apache.paimon.globalindex.GlobalIndexWriter;

import java.io.IOException;

/** Default trainer implementation for centroid-sharded vector global models. */
public class DefaultVectorGlobalModelTrainer implements VectorGlobalModelTrainer {

    private final String indexType;
    private final GlobalIndexWriter indexWriter;
    private final GlobalIndexSingleColumnWriter singleColumnWriter;
    private boolean finished;

    public DefaultVectorGlobalModelTrainer(String indexType, GlobalIndexWriter indexWriter) {
        this.indexType = indexType;
        this.indexWriter = indexWriter;
        this.singleColumnWriter =
                indexWriter instanceof GlobalIndexSingleColumnWriter
                        ? (GlobalIndexSingleColumnWriter) indexWriter
                        : null;
    }

    @Override
    public void write(Object vectorObject, long absoluteRowId) throws IOException {
        if (finished) {
            throw new IllegalStateException("Training has already finished.");
        }
        if (singleColumnWriter != null) {
            singleColumnWriter.write(vectorObject, absoluteRowId);
            return;
        }
        throw new UnsupportedOperationException(
                "Vector global model trainer requires a single-column global index writer for '"
                        + indexType
                        + "', but got "
                        + writerClassName()
                        + ".");
    }

    @Override
    public VectorTrainingModel finishTraining() throws IOException {
        finished = true;
        if (indexWriter instanceof VectorTrainingModelWriter) {
            return ((VectorTrainingModelWriter) indexWriter).finishTraining();
        }
        throw new UnsupportedOperationException(
                "Global index writer "
                        + writerClassName()
                        + " for '"
                        + indexType
                        + "' does not expose a vector training model. ");
    }

    @Override
    public void close() throws IOException {
        if (indexWriter instanceof AutoCloseable) {
            try {
                ((AutoCloseable) indexWriter).close();
            } catch (IOException e) {
                throw e;
            } catch (Exception e) {
                throw new IOException("Failed to close wrapped global index writer.", e);
            }
        }
    }

    private String writerClassName() {
        return indexWriter == null ? "null" : indexWriter.getClass().getName();
    }
}
