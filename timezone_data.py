# Timezone mapping for countries
COUNTRY_TIMEZONES = {
    "USA": "America/New_York",
    "India": "Asia/Kolkata",
    "Brazil": "America/Sao_Paulo",
    "UK": "Europe/London",
    "Canada": "America/Toronto",
    "Germany": "Europe/Berlin",
    "France": "Europe/Paris",
    "Japan": "Asia/Tokyo",
    "Australia": "Australia/Sydney",
}

# YouTube UTC timings
YOUTUBE_UTC_TIMINGS = {
    "USA": {
        "Monday": ["09:00", "13:45", "20:15"],
        "Tuesday": ["09:30", "14:30", "20:45"],
        "Wednesday": ["09:15", "14:00", "20:00"],
        "Thursday": ["09:50", "15:15", "20:30"],
        "Friday": ["10:30", "15:30", "22:00"],
        "Saturday": ["11:45", "16:30", "21:15"],
        "Sunday": ["11:15", "16:00", "21:45"]
    },
    "India": {
        "Monday": ["00:45", "05:15", "13:30"],
        "Tuesday": ["01:10", "06:00", "14:00"],
        "Wednesday": ["00:30", "05:30", "13:45"],
        "Thursday": ["01:00", "06:15", "14:15"],
        "Friday": ["01:30", "07:00", "15:00"],
        "Saturday": ["02:00", "08:00", "14:30"],
        "Sunday": ["02:30", "07:30", "13:45"]
    },
    "Brazil": {
        "Monday": ["08:15", "12:50", "22:00"],
        "Tuesday": ["08:40", "13:30", "21:25"],
        "Wednesday": ["08:00", "13:15", "21:45"],
        "Thursday": ["09:20", "14:00", "22:30"],
        "Friday": ["09:45", "14:30", "23:30"],
        "Saturday": ["10:30", "15:45", "23:15"],
        "Sunday": ["10:00", "15:15", "22:25"]
    },
    "UK": {
        "Monday": ["04:45", "09:15", "16:00"],
        "Tuesday": ["04:20", "09:40", "15:25"],
        "Wednesday": ["05:00", "09:00", "15:45"],
        "Thursday": ["05:30", "09:50", "15:10"],
        "Friday": ["05:15", "09:25", "16:30"],
        "Saturday": ["06:00", "10:45", "16:15"],
        "Sunday": ["06:30", "10:15", "17:00"]
    },
    "Canada": {
        "Monday": ["09:00", "13:45", "20:15"],
        "Tuesday": ["09:30", "14:30", "20:45"],
        "Wednesday": ["09:15", "14:00", "20:00"],
        "Thursday": ["09:50", "15:15", "20:30"],
        "Friday": ["10:30", "15:30", "22:00"],
        "Saturday": ["11:45", "16:30", "21:15"],
        "Sunday": ["11:15", "16:00", "21:45"]
    },
    "Germany": {
        "Monday": ["05:45", "10:15", "17:00"],
        "Tuesday": ["05:20", "10:40", "16:25"],
        "Wednesday": ["06:00", "10:00", "16:45"],
        "Thursday": ["06:30", "10:50", "16:10"],
        "Friday": ["06:15", "10:25", "17:30"],
        "Saturday": ["07:00", "11:45", "17:15"],
        "Sunday": ["07:30", "11:15", "18:00"]
    },
    "France": {
        "Monday": ["05:45", "10:15", "17:00"],
        "Tuesday": ["05:20", "10:40", "16:25"],
        "Wednesday": ["06:00", "10:00", "16:45"],
        "Thursday": ["06:30", "10:50", "16:10"],
        "Friday": ["06:15", "10:25", "17:30"],
        "Saturday": ["07:00", "11:45", "17:15"],
        "Sunday": ["07:30", "11:15", "18:00"]
    },
    "Japan": {
        "Monday": ["06:45", "11:15", "18:00"],
        "Tuesday": ["06:20", "11:40", "17:25"],
        "Wednesday": ["07:00", "11:00", "17:45"],
        "Thursday": ["07:30", "11:50", "17:10"],
        "Friday": ["07:15", "11:25", "18:30"],
        "Saturday": ["08:00", "12:45", "18:15"],
        "Sunday": ["08:30", "12:15", "19:00"]
    },
    "Australia": {
        "Monday": ["07:45", "12:15", "19:00"],
        "Tuesday": ["07:20", "12:40", "18:25"],
        "Wednesday": ["08:00", "12:00", "18:45"],
        "Thursday": ["08:30", "12:50", "18:10"],
        "Friday": ["08:15", "12:25", "19:30"],
        "Saturday": ["09:00", "13:45", "19:15"],
        "Sunday": ["09:30", "13:15", "20:00"]
    },
    "Global": {
        "Monday": ["07:00", "12:00", "18:00"],  # Adjusted by subtracting 2 hours
        "Tuesday": ["07:30", "12:30", "18:30"],
        "Wednesday": ["07:15", "12:15", "18:15"],
        "Thursday": ["07:45", "13:00", "18:45"],
        "Friday": ["08:00", "13:30", "19:00"],
        "Saturday": ["09:00", "14:00", "19:30"],
        "Sunday": ["09:30", "14:30", "19:45"]
    }
}

# Instagram UTC timings
INSTAGRAM_UTC_TIMINGS = {
    "USA": {
        "Monday": ["12:15", "16:45", "23:30"],
        "Tuesday": ["11:50", "17:10", "22:55"],
        "Wednesday": ["12:30", "16:30", "23:15"],
        "Thursday": ["13:00", "17:20", "22:40"],
        "Friday": ["12:45", "16:55", "00:00"],
        "Saturday": ["13:30", "18:15", "23:45"],
        "Sunday": ["14:00", "17:45", "00:30"]
    },
    "India": {
        "Monday": ["03:45", "08:00", "16:50"],
        "Tuesday": ["03:15", "08:30", "16:25"],
        "Wednesday": ["04:00", "07:45", "17:15"],
        "Thursday": ["02:50", "07:20", "16:30"],
        "Friday": ["03:30", "09:00", "17:40"],
        "Saturday": ["04:45", "09:30", "18:00"],
        "Sunday": ["05:15", "10:00", "17:20"]
    },
    "Brazil": {
        "Monday": ["10:45", "15:20", "00:30"],
        "Tuesday": ["11:10", "16:00", "23:55"],
        "Wednesday": ["10:30", "15:45", "00:15"],
        "Thursday": ["11:50", "16:30", "01:00"],
        "Friday": ["12:15", "17:00", "02:00"],
        "Saturday": ["13:00", "18:15", "01:45"],
        "Sunday": ["12:30", "17:45", "00:55"]
    },
    "UK": {
        "Monday": ["07:15", "11:45", "18:30"],
        "Tuesday": ["06:50", "12:10", "17:55"],
        "Wednesday": ["07:30", "11:30", "18:15"],
        "Thursday": ["08:00", "12:20", "17:40"],
        "Friday": ["07:45", "11:55", "19:00"],
        "Saturday": ["08:30", "13:15", "18:45"],
        "Sunday": ["09:00", "12:45", "19:30"]
    },
    "Canada": {
        "Monday": ["12:15", "16:45", "23:30"],
        "Tuesday": ["11:50", "17:10", "22:55"],
        "Wednesday": ["12:30", "16:30", "23:15"],
        "Thursday": ["13:00", "17:20", "22:40"],
        "Friday": ["12:45", "16:55", "00:00"],
        "Saturday": ["13:30", "18:15", "23:45"],
        "Sunday": ["14:00", "17:45", "00:30"]
    },
    "Germany": {
        "Monday": ["06:15", "10:45", "17:30"],
        "Tuesday": ["05:50", "11:10", "16:55"],
        "Wednesday": ["06:30", "10:30", "17:15"],
        "Thursday": ["07:00", "11:20", "16:40"],
        "Friday": ["06:45", "10:55", "18:00"],
        "Saturday": ["07:30", "12:15", "17:45"],
        "Sunday": ["08:00", "11:45", "18:30"]
    },
    "France": {
        "Monday": ["06:15", "10:45", "17:30"],
        "Tuesday": ["05:50", "11:10", "16:55"],
        "Wednesday": ["06:30", "10:30", "17:15"],
        "Thursday": ["07:00", "11:20", "16:40"],
        "Friday": ["06:45", "10:55", "18:00"],
        "Saturday": ["07:30", "12:15", "17:45"],
        "Sunday": ["08:00", "11:45", "18:30"]
    },
    "Japan": {
        "Monday": ["03:15", "07:45", "14:30"],
        "Tuesday": ["02:50", "08:10", "13:55"],
        "Wednesday": ["03:30", "07:30", "14:15"],
        "Thursday": ["04:00", "08:20", "13:40"],
        "Friday": ["03:45", "07:55", "15:00"],
        "Saturday": ["04:30", "09:15", "14:45"],
        "Sunday": ["05:00", "08:45", "15:30"]
    },
    "Australia": {
        "Monday": ["02:15", "06:45", "13:30"],
        "Tuesday": ["01:50", "07:10", "12:55"],
        "Wednesday": ["02:30", "06:30", "13:15"],
        "Thursday": ["03:00", "07:20", "12:40"],
        "Friday": ["02:45", "06:55", "14:00"],
        "Saturday": ["03:30", "08:15", "13:45"],
        "Sunday": ["04:00", "07:45", "14:30"]
    },
    "Global": {
        "Monday": ["10:00", "15:00", "21:00"],
        "Tuesday": ["10:30", "15:30", "21:30"],
        "Wednesday": ["10:15", "15:15", "21:15"],
        "Thursday": ["10:45", "16:00", "21:45"],
        "Friday": ["11:00", "16:30", "22:00"],
        "Saturday": ["12:00", "17:00", "22:30"],
        "Sunday": ["12:30", "17:30", "22:45"]
    }
}