"""
    src / model / link.py
    ---------------------
    The ``Link--State--Predictor`` manages all predictions of the link state 
    based on environmental parameters. States determine whether the communication
    link between the UAV (``Tx-Transmitter``) and the signal antenna
    (``Rx-Receiver``) and whether connections is ``no--link``. This means that
    there is no established connection possible for whatever reason between Rx and
    Tx. It may also predict the connection established is ``No--Line--of--Sight`` 
    (NLOS) suggesting a connection established while lacks free sights, e.g.,
    there is indoor antenna or UAV or otherwise structure is blocking the 
    connection. Finally, the connection could be established as ``Line--of--Sight``
    (LOS) that connects with full free sight, there is full connection indicating
    strong connection. The difference is essentially the ``PSNR`` degrading with
    NLOS compared to the LOS.
    
    The predictions of state of the connection is conducted based upon the relative 
    geometrical distance between the Tx and the Rx where, the angle of departure 
    and angle of arrival suggest the beamforming alter and thus affect the way the 
    established connected is predicted as. Another feature of the predictions of 
    connection is the delay where a longer delay from that the signal is transmitted
    to that it being received also increase the assumption that sight is blocked 
    the longer it takes to pass through, if never receive it increase the assumption
    no connection can exist at all.
"""
from __future__ import annotations

import orjson
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cfg.data import LinkState
from src.cfg.const import TERRESTRIAL, AERIAL

AT = tf.data.AUTOTUNE



class LinkStatePredictor:
    """
    Neural network ``Link--State--Classifier`` predicting the communication 
    conditions between UAV and the antenna. The model performs the actual state 
    conditions and  return its predictions. Predictions of the link state is 
    based on specifically two features 

    - i:    Relative distance vector between any Rx and the Tx
    - ii:   Receiver type wether being on a height or street level

    Internally, receiver type os one-hot encoded into numerical values, while the 
    distance instead given already being numerical is scaled before getting passed
    to the fully connected to the neural network.
    """

    def __init__(self,
        rx_types: List[str], n_unit_links: Tuple[int, ...],
        add_zero_frac_los: float = 0.10, dropout_rate: float = 0.20,
        directory: Union[Path, str] = "link", seed: int = 42
    ):
        """
            Initialize Link--State Predictor Instance
        """
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.model: tfk.Model | None = None
        self.history: tfk.callbacks.History | None = None
        self.rx_encoder: OneHotEncoder | None = None
        self.link_scaler: StandardScaler | None = None

        self.rx_types = rx_types
        self.n_unit_links = n_unit_links
        self.add_zero_los_frac = add_zero_frac_los
        self.dropout_rate = dropout_rate

        self.__version__ = 1
    

    def predict_proba(self, dvec: np.ndarray, rx_types: np.ndarray) -> np.ndarray:
        """
        Predict the probability distribution over all possible link states for 
        each provided receiver--transmitter geometry configurations. This method
        pre--processes the input distance vectors and receiver types using the 
        internally stored scaler and encoder, then forwarding the transformed the
        features through the neural network in inference setting ``(fit=False)``.
        The returned probabilities correspond to the likelihood of each 
        communication condition:

            - No-Link
            - No-Line-of-Sight (NLoS)
            - Line-of-Sight (LoS)
        -----
        Args:
        dvec: Relative distances vectors between the Rx and Tx of shape ``(N, 3)`` \
            where each row correspond to ``[dx, dy, dz]``
        rx_types: Receiver type identifiers associated with each sample. Acceptable \
            representations include mapped numerical identifiers or strings such as \
                "Aerial", or "Terrestrial", "Rx0", "Rx1"
        --------
        Returns:
        Numpy arrays ``np.ndarray`` of shape ``(N, LinkState.n_states)`` containing \
            softmax probabilities for each possible link state.
        -------
        Raises:
        RuntimeError: If the model has not been successfully built at call for this \
            an error is raised for no model can be retained and used.
        """
        if self.model is None:
            raise RuntimeError("Link State Predictor instance has not yet been loaded")
        
        return self.model(self._transform_links(
            np.asarray(dvec, np.float32), np.asarray(rx_types), False
        ), training=False)
    

    def predict_state(self, dvec: np.ndarray, rx_types: np.ndarray) -> tf.Tensor:
        """
        Predict the most likely link -- state class for each provided input sample. 
        This method internally calls the method ``self.predict_proba()`` to obtain 
        the ``softmax`` probabilities distribution across all communication states
        and returns the class index the highest probability. The predicted classes
        corresponding to:
        
            - `LinkState.NO_LINK`
            - `LinkState.LOS`
            - `LinkState.NLOS`
        -----
        Args:
        dvec: Relative distance vectors between transmitter and receiver of shape \
            ``(N, 3)`` where each row correspond ``[dx, dy, dz]``
        rx_types: Receiver type identifier associated with each sample. Acceptable \
            representations include mapped numerical identifier or strings.
        --------
        Returns:
        Tensorflow Tensor of shape ``(N,)`` containing the predicted link--state \
            indices for each sample.
        """
        return tf.argmax(self.predict_proba(dvec, rx_types), axis=-1)
    

    # ============================================================
    #   Model Construction
    # ============================================================

    def build(self):
        """
        Builds the neural network architecture, it build an input layer of twice 
        the  number of input parameter as the size of number of available 
        ``rx_types``,  e.g., ``("AERIAL", "TERRESTRIAL")`` scaled due to 
        polarization of  ``("AOA", "AOD")`` inputs. The input is passed to a
        batch-normalization layer to enhance training performance. The hidden 
        layers are made with a standard dense layer(s), constructed as follows,
        with a ``he-normal`` kernel initializer and a ``l2-regularizer``, each
        followed with its own ``batch-normalization`` and then a standard
        ``sigmoid`` activation.

            - Dense Layer(s)
            - He-Normal kernel initializer as well as a l2-regularization
            - For increased performance, followed by a batch-normalization
            - If a dropout rate is positive, it adds a dropout layer with the rate
            - A sigmoid activation is added final to the layer
        
        The output layer is a output classifier of the various link-states it is
        meant to predict ``LinkState.n_states`` and therefore is activated by a
        standard softmax for multiple classifier outputs.
        """
        inputs = tfk.layers.Input(shape=(2 * len(self.rx_types),), name="Input")
        x = tfk.layers.BatchNormalization()(inputs)

        for i, units in enumerate(self.n_unit_links):
            x = tfk.layers.Dense(
                units = units, kernel_initializer = "he_normal", 
                kernel_regularizer = tfk.regularizers.l2(0.001), name = f"Hidden-{i}"
            )(x)
            x = tfk.layers.BatchNormalization()(x)

            if self.dropout_rate > 0.0:
                x = tfk.layers.Dropout(self.dropout_rate)(x)
            
            x = tfk.layers.Activation("sigmoid")(x)

        outputs = tfk.layers.Dense(
            units = LinkState.n_states, activation="softmax", name="Output"
        )(x)

        self.model = tfk.Model(inputs=inputs, outputs=outputs)
    

    def _prepare_arrays(self,
        data: Dict[str, np.ndarray], fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the raw data--dictionaries into prepared arrays, this change
        the data which is extracted from the ``data``
        """
        dvec = np.asarray(data["dvec"], dtype = np.float32)

        rx = np.asarray(data["rx_type"])
        rx_map = {
            0: TERRESTRIAL, 1: AERIAL, "0": TERRESTRIAL, "1": AERIAL,
            "Terrestrial": TERRESTRIAL, "Aerial": AERIAL,
            "Rx0": TERRESTRIAL, "Rx1": AERIAL, "rx0": TERRESTRIAL, "rx1": AERIAL
        }

        # Ensure Rx values are properly mapped, handling both string and
        # numerical inputs
        rx_str = np.empty_like(rx, dtype = object)
        for i, value in enumerate(rx):

            mapped = rx_map.get(value)
            if mapped is None:
                
                # Try to convert to string if it's numeric
                try:
                    mapped = rx_map.get(str(value))
                
                except:
                    pass
            
            if mapped is None:
                raise ValueError(f"Unknown receiver type: {value}")
            
            rx_str[i] = mapped
        
        link_state = np.asarray(data["link_state"], dtype = np.int32)
        if fit:
            self.rx_encoder = OneHotEncoder(
                categories = [list(self.rx_types)], sparse_output = False,
                handle_unknown = "ignore", dtype = np.float64
            )
            self.rx_encoder.fit(rx_str[:, None])
            self.link_scaler = StandardScaler()
        
        dvec, rx_str, link_state = self._add_los_zero(dvec, rx_str, link_state)
        return self._transform_links(dvec, rx_str, fit), link_state
    

    def _add_los_zero(self,
        dvec: np.ndarray, rx_type: np.ndarray, link_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Augmenting the dataset with synthetic near--zero LoS samples in order
        to avoid undefined behavior which otherwise could occur whenever the 
        UAV ``(Tx)`` gets to close to the present antenna ``(Rx)`` in the 
        simulations.
        """
        n_samples = len(dvec)
        n_add = int(n_samples * self.add_zero_los_frac)

        if n_add <= 0:
            return dvec, rx_type, link_state
        
        idx = np.random.choice(n_samples, size = n_add, replace = True)
        dvec_idx = np.zeros_like(dvec[idx])
        dvec_idx[:, 2] = dvec[idx, 2] # Can this be fixed or need I two rows?

        rx_type_idx, link_state_idx = rx_type[idx], link_state[idx]
        return (
            np.concatenate([dvec, dvec_idx], axis=0),
            np.concatenate([rx_type, rx_type_idx], axis=0),
            np.concatenate([link_state, link_state_idx], axis=0)
        )
    

    def _transform_links(self,
        dvec: np.ndarray, rx_type: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """
        Transform the geometrical link information into a normalized numerical 
        feature representation from the relative distance vector to a more 
        suitable for neural network interference or training. The transformation
        is performed as

        1. Extraction of geometric link feature, total Tx--Rx separation \
            distance, vertical distance component.
        
        2. Receiver type encoding, receiver categories are converted into \
            one-hot encoded representations using the fitted One-Hot-Encoder
        
        3. Feature interaction construction, distance features conditioned on \
            receiver type by element -- wise multiplication between the \
                one - hot receiver representation and the geometric quantities.
        
        4. Feature normalization, standard scaling is applied using the \
            internally store StandardScaler().
        
        The result transformed feature vector is the final representation
        consumed by the neural network classifier.
        -----
        Args:
        dvec: Relative distance Tx-Rx, expecting ``(N,3)`` where each row \
            correspond to ``[dx, dy, dz]``.
        rx_types: Receiver type identifier associated with each sample.
        fit: If ``True``, fits the internal ``StandardScaler`` using the \
            provided dataset before applying normalization.
        --------
        Returns:
        Normalized feature matrix ready for neutral network processing.
        -------
        Raises:
        RuntimeError: If the receiver encoder has not been initialized.
        RuntimeError: If the feature scaler has not been initialized.
        -------
        """
        dr = np.linalg.norm(dvec, axis=1, keepdims=True)
        dh = dvec[:, 2:3]

        if self.rx_encoder is None:
            raise RuntimeError("Encoder not built, call ``_prepare_arrays``")
        
        rx = self.rx_encoder.transform(rx_type[:, None]).astype(np.float32)
        x = np.hstack([rx * dr, rx * dh])

        if self.link_scaler is None:
            raise RuntimeError(
                "``link_scaler`` is not initialized ``_prepare_arrays``"
            )
        
        return self.link_scaler.fit_transform(x) if fit \
            else self.link_scaler.transform(x)

    # ============================================================
    #   Model Fitting
    # ============================================================

    def fit(self,
        dtr: Dict[str, np.ndarray], dts: Dict[str, np.ndarray],
        epochs: int = 50, size: int = 512, learning_rate: float = 1e-3
    ) -> tfk.callbacks.History:
        """
        """
        xtr, ytr = self._prepare_arrays(dtr, True)
        xts, yts = self._prepare_arrays(dts, False)

        self.model.compile(
            optimizer=tfk.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        t = tf.data.Dataset.from_tensor_slices((xtr,ytr)).batch(size).prefetch(AT)
        v = tf.data.Dataset.from_tensor_slices((xts,yts)).batch(size).prefetch(AT)
        
        history = self.model.fit(
            t, epochs=epochs, validation_data=v, verbose=1
        )
        self.history = history
        return history
    
    # ============================================================
    #   I/O -- Saving / Loading
    # ============================================================

    # def save(self):
    #     """
    #     Saving the model to preserve the model. It includes all features of the 
    #     model and weights whether its being trained.

    #         1. Model saves metadata and configurations (JSON)
    #             - library and model versioning information
    #             - architecture, related configuration parameters
    #             - optional training history

    #         2. Tensorflow saves the model weight (Tensorflow)
            
    #         3. Serializes the scikit-learn preprocessing object requiring
    #            for inference and encoders, stored in separate files.
        
    #     The preprocessing objects are serialized using a lightweight schema that 
    #     captures only the state needed for inference, avoiding the pickle / job 
    #     inference of scalars to ensure portability between different types of 
    #     platforms.
    #     """
    #     from src.models.utils.preproc import serialize_preproc
    #     with open(self.dir / LINK_CONFIG_FN, "wb") as fp:
    #         fp.write(orjson.dumps({
    #             "version": self.__version__, 
    #             "framework": { "tensorflow": tf.__version__ },
    #             "config": {
    #                 "rx_types": self.rx_types,
    #                 "n_unit_links": self.n_unit_links,
    #                 "add_zero_los_frac": self.add_zero_los_frac,
    #                 "dropout_rate": self.dropout_rate
    #             },
    #             "history": getattr(self.history, "history", None) if self.history \
    #                 else None
    #         }, option=orjson.OPT_INDENT_2))
        
    #     self.model.save_weights(str(self.dir / WEIGHTS_FN))
    #     proc: Dict[str, Any] = {}

    #     if self.link_scaler:
    #         proc["link_scaler"] = serialize_preproc(self.link_scaler)
        
    #     if self.rx_encoder:
    #         proc["rx_encoder"] = serialize_preproc(self.rx_encoder)
        
    #     with open(self.dir / PREPROC_FN, "wb") as fp:
    #         fp.write(orjson.dumps(proc, option=orjson.OPT_INDENT_2))
    

    # def load(self):
    #     """
    #     Restore a previously saved model state from secondary memory to 
    #     retrieve the model parameters, architecture, potential weights:

    #         - Loads model metadata and configuration and applies it to
    #           the current instance, emitting a warning if the saved model
    #           version does match the current code version.

    #         - Reconstruct serialized scikit-learn preprocessing objects 
    #           required for inference.
            
    #         - Rebuilds the tensorflow model architecture and loads the 
    #           saved weights.
            
    #         - Restore training history if it was saved.
        
    #     The model is fully ready for inference after the call. Any 
    #     existing model, preprocessing, or configurations on the instance
    #     are overwritten by the loaded state.
    #     """
    #     from src.models.utils.preproc import deserialize_preproc
    #     with open(self.dir / LINK_CONFIG_FN, "rb") as fp:
    #         payload = orjson.loads(fp.read())
        
    #     if payload.get("version", 0) != self.__version__:
    #         print(
    #             f"Warning: Version mismatch. Model: {payload.get('version')},"
    #             f"current: `{self.__version__}`"
    #         )
        
    #     config = payload.get("config", {})
    #     self.rx_types = config.get("rx_types", self.rx_types)
    #     self.n_unit_links = tuple(config.get("n_unit_links", self.n_unit_links))
    #     self.add_zero_los_frac = float(config.get("add_zero_los_frac", self.add_zero_los_frac))
    #     self.dropout_rate = float(config.get("dropout_rate", self.dropout_rate))
        
    #     with open(self.dir / PREPROC_FN, "rb") as fp:
    #         preproc_dict = orjson.loads(fp.read())
        
    #     self.link_scaler = deserialize_preproc(preproc_dict["link_scaler"])
    #     self.rx_encoder = deserialize_preproc(preproc_dict["rx_encoder"])
        
    #     self.build()
    #     self.model.load_weights(str(self.dir / WEIGHTS_FN))

    #     if payload.get("history"):
    #         self.history = tfk.callbacks.History()
    #         self.history.history = payload["history"]
        
    #     print(f"Model loaded from {self.dir}")

