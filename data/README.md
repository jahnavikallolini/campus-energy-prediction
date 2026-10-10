# Raw data: I-BLEND (Indian BuiLdings Energy coNsumption Dataset)

Source: H. Rashid, P. Singh and A. Singh, "I-BLEND, a campus-scale commercial and residential buildings
electrical energy dataset," *Scientific Data* 6, 190015 (2019), doi:10.1038/sdata.2019.15
(figshare collection doi:10.6084/m9.figshare.c.3893581). Raw files are never modified by this project.

## Expected layout (`data/raw/`, or set the `IBLEND_RAW_DIR` environment variable)

```
data/raw/
  energy/
    acad_build_mains.csv  boys_hostel_mains.csv  facilities_build_mains.csv  girls_hostel_mains.csv
    lecture_build_mains.csv  library_build_mains.csv  mess_build_mains.csv      <- the seven main feeders
    all_buildings_power.csv                                                     <- audit / backup feeders only
    data_present_status_buildings.csv  data_present_status_transformers.csv     <- audit only
  occupancy/   ACB.csv BH.csv DB.csv GH.csv LB.csv LCB.csv SRB.csv   (unzip IIITD_occupancy_dataset.zip)
  calendar/    calender_year_2013_.csv ... calender_year_2017_.csv     (unzip iiitd_calender_schedule.zip)
  weather/     IIITD_and_airport_data.csv                              (unzip weather_comparison.zip)
```

## What the files contain (verified from the files; units and code meanings from the Data Descriptor)

| File group | Schema | Frequency | Range |
|---|---|---|---|
| Building mains | `timestamp, power, current, voltage, frequency, power_factor` | 1 minute | 2013-08-10 to 2017-12-31 (Facilities from 2013-11-15, Mess from 2013-09-24) |
| `all_buildings_power.csv` | `timestamp` + 9 power columns (7 mains + `Boys_backup`, `Girls_backup`) | 1 minute | 2013-08-10 to 2017-12-31 |
| Status files | text timestamp + 0/1 "data present" flags | 1 minute | same grid |
| Occupancy | `timestamp, occupancy_count` | 10 minutes | 2014-02-16 to 2017-11-03 (Facilities/SRB from 2014-07-10) |
| Calendar | `Date, working_day, activity` | daily | 2013-08-01 to 2017-12-31 |
| Weather | datetime index, `Airport_temperature`, `Airport_Humidity`, `IIITD_temperature`, `IIITD_Humidity` | 30 minutes | **2018-03-01 to 2018-06-29** |

* `timestamp` is Unix epoch seconds (UTC); convert with Asia/Kolkata (+05:30). Processed tables use tz-naive IST wall-clock time.
* `power` is in **watts** (Data Descriptor: "power (watts)"). Building meters are Schneider EM6400 panel meters.
* Occupancy is a Wi-Fi-derived device count; the released 10-minute values are the **maximum** of the one-minute counts in each window. It over-counts people with several devices and under-counts people without Wi-Fi.
* Calendar `activity`: `H` = high-activity period (academic semester days with students in dormitories); `L` = low-activity period (semester, summer, winter and festival breaks). `working_day` is the institute's working / non-working flag.
* Building codes: Academic=ACB, Boys hostel=BH, Mess (Dining)=DB, Girls hostel=GH, Library=LB, Lecture=LCB, Facilities=SRB.
* Each dormitory has a mains meter and a separate UPS ("backup") meter. The primary seven-building analysis uses only the **mains** feeders, as the synopsis defines seven buildings; `Boys_backup` and `Girls_backup` are documented but excluded.
* Missing minutes are simply absent rows in the building files (`all_buildings_power.csv` shows them as `NA`).
* The weather files cover March-June 2018, after the last energy reading, so weather cannot be aligned with the energy data and is excluded from the models.
