# Timestamp discontinuity (July 2017): investigation summary

Status: **unresolved**. Raw data and processed data were not modified; no correction was applied.

Evidence (all reproducible: `python -m src.analysis.timestamp_shift_diagnostic`, `python -m src.analysis.timestamp_offset_by_month`)
1. Monthly daily-profile alignment against a 2015-08..2016-06 reference is 0 h (within about 2 h) for 2013-08..2017-06 and 4-6 h for 2017-07..2017-12 in all seven energy series (`timestamp_shift_monthly_profile_shift_hours.csv`, `fig19_timestamp_shift_profiles.png`).
2. The change is abrupt, between 2017-07-07 and 2017-07-10, directly after long logging gaps (Academic and Lecture: 742 min on 2017-07-08 12:21; Mess: 971 min at 16:10; Library: 33 h on 2017-07-10).
3. Occupancy timestamps do not shift (all seven occupancy profiles remain within about 1 h of the reference).
4. Against occupancy, the best energy offset is 0 to 70 min in January-June 2017 and 250 to 330 min in July-October 2017 (`timestamp_offset_by_month.csv`). The post-break offset is not identical across buildings (Academic about 300-330 min, Boys Hostel about 250-290 min), so a single exact value (for example 330 min = 5 h 30 min) cannot be established from the data.
5. The dataset authors' reader script (i-blend.github.io, `py_scripts/read_energy_occupancy.py`) adds a fixed +5:30 offset to the epoch timestamps for all periods and does not mention a break. The conversion used here is the documented one. I found no documentation of a clock change, and I could not access the original figshare files or an erratum from this environment.

Decision (as instructed when unresolved): primary test period = 2017-01-01 to 2017-07-07; the original full test period (to 2017-11-04) and the post-break part (2017-07-11 onward) are reported separately as sensitivity analyses. 2017-07-08..07-10 is treated as a transition. Models, hyperparameters and the validation-based model selection are unchanged (validation ends 2016-12-31 and is unaffected).
