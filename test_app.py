import pandas as pd
import pytest
from modules.data_processing import read_csv_bytes, validate_dataset, build_prediction_frame, MODEL_FEATURES
from modules.dashboard import dashboard_kpis, review_queue
from modules.prediction import load_model

def test_valid_csv_upload():
    frame = read_csv_bytes(b"Request_ID,Queue_Length\nSR1,2\n")
    assert frame.shape == (1, 2)

@pytest.mark.parametrize("contents", [b"", b"  ", b"a,b\n"])
def test_empty_csv_errors(contents):
    with pytest.raises(ValueError):
        read_csv_bytes(contents)

def test_malformed_csv_errors():
    # pandas may parse ragged rows depending on the shape; this deliberately malformed quote is rejected.
    with pytest.raises(ValueError):
        read_csv_bytes(b'a,b\n"unterminated,2\n')

def test_missing_required_columns_reported():
    result = validate_dataset(pd.DataFrame({"Request_ID":["SR1"]}))
    assert "Service_Type" in result["missing_columns"]
    assert result["valid_schema"] is False

def test_missing_values_and_duplicates_reported():
    frame = pd.DataFrame({"Request_ID":["SR1","SR1"],"Service_Type":["Records",None]})
    result = validate_dataset(frame)
    assert result["duplicate_request_ids"] == 1
    assert result["missing_values"]["Service_Type"] == 1

def test_prediction_feature_order_and_numeric_types():
    values = dict(Service_Type="Records",Requester_Type="Student",Requested_Channel="Portal",Priority="Normal",
        Assigned_Office="Student Affairs",Queue_Length="4",Queue_Category="Low",Required_Documents_Complete="Yes",
        SLA_Days="5",Submission_Weekday="Monday")
    frame = build_prediction_frame(values)
    assert list(frame.columns) == MODEL_FEATURES
    assert frame.loc[0,"Queue_Length"] == 4.0
    assert frame.loc[0,"SLA_Days"] == 5.0

def test_invalid_prediction_inputs():
    values = dict(Service_Type="Records",Requester_Type="Student",Requested_Channel="Portal",Priority="Normal",
        Assigned_Office="Student Affairs",Queue_Length=-1,Queue_Category="Low",Required_Documents_Complete="Yes",
        SLA_Days=5,Submission_Weekday="Monday")
    with pytest.raises(ValueError):
        build_prediction_frame(values)

def test_dashboard_calculations():
    frame = pd.DataFrame({"Status":["Completed","Pending","Completed"],"SLA_Breached":[1,None,0]})
    result = dashboard_kpis(frame)
    assert result["total_requests"] == 3
    assert result["completed_requests"] == 2
    assert result["labeled_requests"] == 2
    assert result["observed_breach_rate_percent"] == 50.0

def test_review_queue_does_not_expose_satisfaction():
    frame = pd.DataFrame({"Request_ID":["SR1"],"Service_Type":["Records"],"Satisfaction_Score":[1]})
    assert "Satisfaction_Score" not in review_queue(frame).columns

def test_missing_model_reports_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.joblib"))
