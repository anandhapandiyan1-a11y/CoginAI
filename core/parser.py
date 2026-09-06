import os
import pandas as pd


class DataParser:

    def __init__(self, file_source):
        """Initialize parser with a file path, uploaded file buffer, or BytesIO object."""
        self.file_source = file_source
        self.filename = getattr(file_source, "name", str(file_source))

    def _get_file_extension(self) -> str:
        """Extract file extension."""
        _, ext = os.path.splitext(self.filename)
        return ext.lower()

    def load_data(self) -> pd.DataFrame:
        """Automated multi-format file loader with encoding resilience."""
        ext = self._get_file_extension()

        try:
            if ext == ".csv":
                try:
                    df = pd.read_csv(self.file_source, encoding="utf-8")
                except UnicodeDecodeError:
                    if hasattr(self.file_source, "seek"):
                        self.file_source.seek(0)
                    df = pd.read_csv(self.file_source, encoding="latin1")

            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(self.file_source)

            elif ext == ".json":
                df = pd.read_json(self.file_source)

            elif ext == ".parquet":
                df = pd.read_parquet(self.file_source)

            else:
                raise ValueError(
                    f"Unsupported file format '{ext}'. Please upload CSV, Excel, JSON, or Parquet."
                )

            df = self._clean_column_names(df)
            df = self._auto_parse_datetimes(df)

            return df

        except Exception as e:
            raise RuntimeError(f"Error parsing file '{self.filename}': {str(e)}")

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column headers."""
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace(r"[^\w\s]", "", regex=True)
        )
        return df

    def _auto_parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects and converts date/time columns."""
        for col in df.columns:
            if df[col].dtype == "object":
                col_lower = col.lower()
                if any(
                    key in col_lower
                    for key in ["date", "time", "timestamp", "year", "month"]
                ):
                    try:
                        df[col] = pd.to_datetime(df[col], errors="ignore")
                    except Exception:
                        pass
        return df