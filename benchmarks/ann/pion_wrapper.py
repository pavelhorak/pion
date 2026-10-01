import numpy
from ann_benchmarks.algorithms.base import BaseANN
import redis

class PionANN(BaseANN):
    def __init__(self, metric, index_params):
        if metric not in ('euclidean', 'ip'):
            raise NotImplementedError(f"Pion doesn't support metric {metric}")
        self._metric = metric
        self._index_params = index_params
        
        # Connection parameters
        self._host = '127.0.0.1'
        self._port = 1974
        self._r = None
        self._index_name = "pion-ann-index"

    def fit(self, X):
        """
        Build the index for the data points given in X.
        X is a numpy array of size n*d, where n is the number of points and d is the dimension.
        """
        if self._r is None:
            self._r = redis.Redis(host=self._host, port=self._port)
        
        # Ensure the server is reachable
        self._r.ping()
        
        # Flush any old data
        try:
            self._r.ft(self._index_name).dropindex()
        except redis.exceptions.ResponseError:
            # Index might not exist, which is fine
            pass

        # Pion HNSW parameters
        m = self._index_params.get("M", 16)
        ef_construction = self._index_params.get("efConstruction", 100)
        
        # Define the schema for the vector index
        schema = (
            redis.ft.VectorField("vec", "HNSW", {"TYPE": "FLOAT32", "DIM": X.shape[1], "DISTANCE_METRIC": "L2" if self._metric == 'euclidean' else "IP", "M": m, "EF_CONSTRUCTION": ef_construction}),
        )
        
        # Create the index
        self._r.ft(self._index_name).create_index(schema)

        # Use a pipeline to insert all vectors
        pipe = self._r.pipeline(transaction=False)
        for i, vec in enumerate(X):
            key = f"doc:{i}"
            # Pion uses HSET to store vectors for FT.SEARCH
            pipe.hset(key, mapping={"vec": vec.astype(numpy.float32).tobytes()})
            if i % 1000 == 0:
                pipe.execute() # Execute in batches
        pipe.execute()

    def set_query_arguments(self, ef):
        """
        Set the query-time parameter, ef.
        """
        self._ef = ef

    def query(self, v, n):
        """
        Query the index for the n closest neighbors of vector v.
        """
        query_vector = v.astype(numpy.float32).tobytes()
        
        # Create the query. Pion's FT.SEARCH expects a query like "*=>[KNN $K @vec $BLOB]".
        q = (
            redis.ft.Query("*=>[KNN $K @vec $BLOB]")
            .sort_by("__vec_score")
            .return_fields("id")
            .dialect(2)
        )
        
        query_params = {
            "K": n,
            "BLOB": query_vector,
        }
        
        results = self._r.ft(self._index_name).search(q, query_params)
        
        # The result objects have a 'id' attribute which corresponds to the key.
        # We need to parse the integer index from "doc:XYZ".
        return [int(doc.id.split(':')[1]) for doc in results.docs]

    def get_memory_usage(self):
        """
        Return the memory usage of the index in bytes.
        Pion might expose this via an INFO or custom command.
        For now, we return 0 as a placeholder.
        """
        # info = self._r.info()
        # return info.get('used_memory', 0)
        return 0

    def __str__(self):
        return f"PionANN(M={self._index_params.get('M')}, efConstruction={self._index_params.get('efConstruction')})"

    def done(self):
        if self._r:
            self._r.close()

