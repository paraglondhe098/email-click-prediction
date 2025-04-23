from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List, Callable, Union, Any, Optional
import pandas as pd
import numpy as np


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies specified transformers to columns of a DataFrame.

    Parameters
    ----------
    transformers : List[Tuple[str, BaseEstimator, List[str]]]
        A list of Tuples where first entry is transformer name, and second and third entries are
        a transformer instance and a list of column names to which it should be applied.
    """

    def __init__(self, transformers: List[Tuple[str, BaseEstimator, List[str]]]):
        self.transformers = transformers

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Fits all transformers on the corresponding columns of the DataFrame.
        """

        self._validate_input(X)
        for name, trf, cols in self.transformers:
            trf.fit(X[cols])
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """
        Transforms the DataFrame using the fitted transformers.
        """
        self._validate_input(X)
        X = X.copy()
        for name, trf, cols in self.transformers:
            if isinstance(trf, OneHotEncoder):
                X = self._apply_one_hot_encoding(X, trf, cols)
            else:
                X[cols] = trf.transform(X[cols])
        return X

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params) -> pd.DataFrame:
        """
        Fits the transformers and transforms the DataFrame in a single step.
        """
        self._validate_input(X)
        X = X.copy()
        for name, trf, cols in self.transformers:
            if isinstance(trf, OneHotEncoder):
                X = self._apply_one_hot_encoding(X, trf, cols, fit=True)
            else:
                X[cols] = trf.fit_transform(X[cols])
        return X

    def _apply_one_hot_encoding(self, X: pd.DataFrame, encoder: OneHotEncoder, cols: List[str],
                                fit=False) -> pd.DataFrame:
        """
        Handles one-hot encoding and merges the transformed data back into the DataFrame.
        """
        if fit:
            transformed = encoder.fit_transform(X[cols])
        else:
            transformed = encoder.transform(X[cols])

        if encoder.sparse_output:
            transformed = transformed.toarray()

        # Generate column names
        if encoder.drop:
            cats = [
                f"{cols[i]}_{cat}" for i, catset in enumerate(encoder.categories_)
                for cat in np.delete(catset, encoder.drop_idx_[i])
            ]
        else:
            cats = [f"{cols[0]}_{cat}" for cat in np.hstack(encoder.categories_)]

        onehot_df = pd.DataFrame(transformed, columns=cats, index=X.index)
        return pd.concat([X.drop(columns=cols), onehot_df], axis=1)

    def _validate_input(self, X: pd.DataFrame):
        """
        Validates the input DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        for name, _, cols in self.transformers:
            missing_cols = [col for col in cols if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for transformer '{name}': {missing_cols}")


class TransformerLambda(BaseEstimator, TransformerMixin):
    def __init__(self, func: Callable[[Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]):
        self.func = func

    def fit(self, X: Union[pd.DataFrame, pd.Series], y=None):
        """
        Fits all transformers on the corresponding columns of the DataFrame.
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        return self.func(X)

    def fit_transform(self, X: Union[pd.DataFrame, pd.Series], y=None, **fit_params) -> Union[
        pd.DataFrame, pd.Series]:
        return self.func(X)

    # class CustomTransformerDict(BaseEstimator, TransformerMixin):
    #     """
    #     A custom transformer that applies specified transformers to columns of a DataFrame.
    #
    #     Parameters
    #     ----------
    #     transformers : Dict[str, Tuple[BaseEstimator, List[str]]]
    #         A dictionary where keys are transformer names, and values are tuples containing
    #         a transformer instance and a list of column names to which it should be applied.
    #     """
    #
    #     def __init__(self, transformers: Dict[str, Tuple[BaseEstimator, List[str]]]):
    #         self.transformers = transformers
    #
    #     def fit(self, X: pd.DataFrame, y = None):
    #         """
    #         Fits all transformers on the corresponding columns of the DataFrame.
    #         """
    #
    #         self._validate_input(X)
    #         for name, (trf, cols) in self.transformers.items():
    #             trf.fit(X[cols])
    #         return self
    #
    #     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    #         """
    #         Transforms the DataFrame using the fitted transformers.
    #         """
    #         self._validate_input(X)
    #         X = X.copy()
    #         for name, (trf, cols) in self.transformers.items():
    #             if isinstance(trf, OneHotEncoder):
    #                 X = self._apply_one_hot_encoding(X, trf, cols)
    #             else:
    #                 X[cols] = trf.transform(X[cols])
    #         return X
    #
    #     def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
    #         """
    #         Fits the transformers and transforms the DataFrame in a single step.
    #         """
    #         self._validate_input(X)
    #         X = X.copy()
    #         for name, (trf, cols) in self.transformers.items():
    #             if isinstance(trf, OneHotEncoder):
    #                 X = self._apply_one_hot_encoding(X, trf, cols, fit=True)
    #             else:
    #                 X[cols] = trf.fit_transform(X[cols])
    #         return X
    #
    #     def _apply_one_hot_encoding(self, X: pd.DataFrame, encoder: OneHotEncoder, cols: List[str], fit=False) -> pd.DataFrame:
    #         """
    #         Handles one-hot encoding and merges the transformed data back into the DataFrame.
    #         """
    #         if fit:
    #             transformed = encoder.fit_transform(X[cols])
    #         else:
    #             transformed = encoder.transform(X[cols])
    #
    #         if encoder.sparse_output:
    #             transformed = transformed.toarray()
    #
    #         # Generate column names
    #         if encoder.drop:
    #             cats = [
    #                 f"{cols[i]}_{cat}" for i, catset in enumerate(encoder.categories_)
    #                 for cat in np.delete(catset, encoder.drop_idx_[i])
    #             ]
    #         else:
    #             cats = [f"{cols[0]}_{cat}" for cat in np.hstack(encoder.categories_)]
    #
    #         onehot_df = pd.DataFrame(transformed, columns=cats, index=X.index)
    #         return pd.concat([X.drop(columns=cols), onehot_df], axis=1)
    #
    #     def _validate_input(self, X: pd.DataFrame):
    #         """
    #         Validates the input DataFrame.
    #         """
    #         if not isinstance(X, pd.DataFrame):
    #             raise ValueError("Input must be a pandas DataFrame.")
    #         for name, (_, cols) in self.transformers.items():
    #             missing_cols = [col for col in cols if col not in X.columns]
    #             if missing_cols:
    #                 raise ValueError(f"Missing columns for transformer '{name}': {missing_cols}")
