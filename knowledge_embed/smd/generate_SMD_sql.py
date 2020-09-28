def generate_weather_sql(key_group, constraints):
    poi_num = 1
    sql_cols = ["location", "date"]
    for key in key_group:
        poi_num = max([int(max(key_group[key])), poi_num])
        key = key.replace("temperature_", "").replace("_attribute", "").replace("@", "")
        if key != "date" and key != "location":
            sql_cols.append(key)

    sql = ["SELECT DISTINCT"]
    # weekly negation
    if "negation" in constraints and constraints["negation"] == 0:
        # SELECT loc, Nweather FROM (SELECT DISTINCT c1.location as loc , c8.weather as Nweather FROM content AS c1 JOIN content AS c2 INNER JOIN content AS c3 JOIN content AS c4 JOIN content AS c5 JOIN content AS c6 JOIN content AS c7 JOIN content AS c8 ON (c7.date > c6.date and c6.date > c5.date and c5.date> c4.date and c4.date> c3.date and c3.date> c2.date and c2.date > c1.date and c1.location == c2.location and c2.location == c3.location and c2.location == c3.location and c2.location == c3.location and c3.location == c4.location and c4.location == c5.location and c5.location == c6.location and c6.location == c7.location and c8.weather != c7.weather and c8.weather != c6.weather and c8.weather != c5.weather and c8.weather != c4.weather and c8.weather != c3.weather and c8.weather != c2.weather and c8.weather != c1.weather))
        sql_N_content = "(SELECT DISTINCT c1.location as loc , c8.weather as Nweather FROM content AS c1 JOIN content AS c2 INNER JOIN content AS c3 JOIN content AS c4 JOIN content AS c5 JOIN content AS c6 JOIN content AS c7 JOIN content AS c8 ON (c7.date > c6.date and c6.date > c5.date and c5.date> c4.date and c4.date> c3.date and c3.date> c2.date and c2.date > c1.date and c1.location == c2.location and c2.location == c3.location and c2.location == c3.location and c2.location == c3.location and c3.location == c4.location and c4.location == c5.location and c5.location == c6.location and c6.location == c7.location and c8.weather != c7.weather and c8.weather != c6.weather and c8.weather != c5.weather and c8.weather != c4.weather and c8.weather != c3.weather and c8.weather != c2.weather and c8.weather != c1.weather))"
        if poi_num == 1 and not "max_min" in constraints and not "max_min_week" in constraints:
            if "@date" not in key_group and ("@weekly_time" not in constraints or ("@weekly_time" in constraints and "1" not in constraints["@weekly_time"] and "2" not in constraints["@weekly_time"])):
                sql_cols.remove("date")
                sql_cols.remove("weather")

            sql.append("d1.Nweather,")
            sql.append("d2."+", d2.".join(sql_cols))
            sql.append(f"FROM {sql_N_content} AS d1 JOIN content AS d2")
            sql_constraint = []
            sql_constraint.append("d1.loc == d2.location")
            if "@weekly_time" in constraints and "1" in constraints["@weekly_time"]: # today
                sql_constraint.append("date == '1monday'")
            elif "@weekly_time" in constraints and "2" in constraints["@weekly_time"]: # tmr
                sql_constraint.append("date == '2tuesday'")
            elif "@weekly_time" in constraints and "3" in constraints["@weekly_time"]: # weekend
                sql_constraint.append("(date == '6saturday' or date == '7sunday')")
            elif "@today" in constraints: # today
                sql_constraint.append("date == '1monday'")
            if len(sql_constraint) > 0:
                sql.append("WHERE "+" and ".join(sql_constraint))
        elif poi_num > 1 and not "max_min" in constraints and not "max_min_week" in constraints:
            sql.append("d1.Nweather, d2.location,")
            sql_content = [f"FROM {sql_N_content} AS d1 JOIN"]
            sql_constraint  = []
            sql_constraint.append("d1.loc == d2.location")
            for i in range(2, poi_num+2):
                for item in sql_cols[1:]:
                    sql.append(f"d{i}."+item+",")

                if i < poi_num+1:
                    sql_constraint.append(f"d{i+1}.date > d{i}.date")
                    sql_constraint.append(f"d{i+1}.location == d{i}.location")
                sql_content.append(f"content AS d{i}")
                if i != poi_num+1:
                    sql_content.append("JOIN")
            if "@weekly_time" in constraints and "1" in constraints["@weekly_time"]: # today
                sql_constraint.append(f"d{2}.date == '1monday'")
            if "@weekly_time" in constraints and "2" in constraints["@weekly_time"]: # tmr
                assert poi_num == 2
                sql_constraint.append(f"d{3}.date == '2tuesday'")
            if "@weekly_time" in constraints and "3" in constraints["@weekly_time"]: # weekend
                assert poi_num == 2
                sql_constraint.append(f"d{2}.date == '6saturday'")
                sql_constraint.append(f"d{3}.date == '7sunday'")

            sql[-1] = sql[-1][:-1] # remove ","
            sql = sql + sql_content + ["ON ("] + [" and ".join(sql_constraint)] + [")"]
        else:
            raise ValueError("Invalid dialog template type!")
        return " ".join(sql), ["weather_attribute_10"]+sql_cols
    elif "negation" in constraints and constraints["negation"] == 1: # daily negation
        raise ValueError("Invalid dialog template type!")
    
    # not negation
    if poi_num == 1 and not "max_min" in constraints and not "max_min_week" in constraints:
        sql.append(", ".join(sql_cols))
        sql.append("FROM content")
        sql_constraint = []
        if "@weekly_time" in constraints and "1" in constraints["@weekly_time"]: # today
            sql_constraint.append("date == '1monday'")
        elif "@weekly_time" in constraints and "2" in constraints["@weekly_time"]: # tmr
            sql_constraint.append("date == '2tuesday'")
        elif "@today" in constraints: # today
            sql_constraint.append("date == '1monday'")
        if len(sql_constraint) > 0:
            sql.append("WHERE "+" and ".join(sql_constraint))
    elif poi_num == 1 and "max_min" in constraints:
        # select date first
        sql_constraint  = ["(SELECT DISTINCT"]
        sql_constraint.append(", ".join(sql_cols))
        sql_constraint += ["FROM content WHERE"]
        for i in contraints["@weekly_time"]:
            if i == 1:
                sql_constraint += ["date == '1monday'"]
                sql_constraint += ["or"]
            if i == 2:
                sql_constraint += ["date == '2tuesday'"]
                sql_constraint += ["or"]
            if i == 3:
                sql_constraint += ["date == '6saturday' or date == '7sunday'"]
                sql_constraint += ["or"]
        sql_constraint[-1] = ")" # remove one more "or"

        cols = []
        for col in sql_cols:
            if "@temperature_low_1" in constraints and col == "low":
                cols.append(f"MIN(low)")
            elif "@temperature_high_1" in constraints and col == "high":
                cols.append(f"MAX(high)")
            else:
                cols.append(col)
        sql.append(", ".join(cols))
        sql.append(f"FROM {' '.join(sql_constraint)}")
        sql.append("GROUP BY location")
    elif poi_num == 1 and "max_min_week" in constraints:
        cols = []
        for col in sql_cols:
            if "@temperature_low_1" in constraints and col == "low":
                cols.append(f"MIN(low)")
            elif "@temperature_high_1" in constraints and col == "high":
                cols.append(f"MAX(high)")
            else:
                cols.append(col)
        sql.append(", ".join(cols))
        sql.append("FROM content")
        sql.append("GROUP BY location")
    elif poi_num > 1 and "max_min" in constraints: # in constraints: 
        # SELECT DISTINCT c1.location, c3.date, c3.weather, c1.Mlow, c2.Mhigh, c4.date, c4.weather, c4.low, c4.high FROM (SELECT DISTINCT location, date, MIN(low) AS Mlow FROM content WHERE date == '1monday' or date == '2tuesday' GROUP BY location) AS c1 JOIN (SELECT DISTINCT location, date, MAX(high) AS Mhigh FROM content WHERE date == '1monday' or date == '2tuesday' GROUP BY location) AS c2 JOIN content AS c3 JOIN content AS c4 ON (c1.location == c2.location and c2.location == c3.location and c3.location == c4.location and c3.date == '1monday' and c4.date == '2tuesday')
        sql_content = ["FROM"]
        sql_constraint = []
        max_min_num = 2 if "@temperature_high_1" in constraints and "@temperature_low_1" in constraints else 1

        sql.append("c1."+sql_cols[0]+",")
        max_min_cursor = 1

        # build inner constraints: 
        MAXi, MINi = 0, 0
        for i in range(1, max_min_num+1):
            inner_content = ["SELECT DISTINCT"]
            for item in sql_cols:
                if item == "high" and "@temperature_high_1" in constraints and MAXi == 0 and MINi != i:
                    inner_content.append(f"MAX(high) AS Mhigh,")
                    MAXi = i
                elif item == "low" and "@temperature_low_1" in constraints and MAXi != i and MINi == 0:
                    inner_content.append(f"MIN(low) AS Mlow,")
                    MINi= i
                elif item == "location" or item == "date":
                    inner_content.append(f"{item},")
            inner_content[-1] = inner_content[-1][:-1]
            inner_content.append("FROM content WHERE")
            cc = []
            if "@weekly_time" in constraints and "1" in constraints["@weekly_time"]: # today
                cc.append(f"date == '1monday'")
            if "@weekly_time" in constraints and "2" in constraints["@weekly_time"]: # tmr
                assert poi_num == 2
                cc.append(f"date == '2tuesday'")
            if "@weekly_time" in constraints and "3" in constraints["@weekly_time"]: # weekend
                assert poi_num == 2
                cc.append(f"date == '6saturday'")
                cc.append(f"date == '7sunday'")
            inner_content.append(" or ".join(cc))
            inner_content.append("GROUP BY location")
            sql_content.append(f"({' '.join(inner_content)}) AS c{i} JOIN")

        for i in range(1, poi_num+1):
            for item in sql_cols[1:]:
                if item == "high" and "@temperature_high_1" in constraints and i == 1 and max_min_cursor <= max_min_num:
                    sql.append(f"c{max_min_cursor}.Mhigh,")
                    max_min_cursor += 1
                elif item == "low" and "@temperature_low_1" in constraints and i == 1 and max_min_cursor <= max_min_num:
                    sql.append(f"c{max_min_cursor}.Mlow,")
                    max_min_cursor += 1
                else:
                    sql.append(f"c{i+max_min_num}.{item},")

            if i > 1:
                sql_constraint.append(f"c{i}.location == c{i-1}.location")
            sql_constraint.append(f"c{i+max_min_num}.location == c{i+max_min_num-1}.location")
            sql_content.append(f"content AS c{i+max_min_num}")
            if i != poi_num:
                sql_content.append("JOIN")
        
        if "@weekly_time" in constraints and "1" in constraints["@weekly_time"]: # today
            sql_constraint.append(f"c{1+max_min_num}.date == '1monday'")
        if "@weekly_time" in constraints and "2" in constraints["@weekly_time"]: # tmr
            assert poi_num == 2
            sql_constraint.append(f"c{2+max_min_num}.date == '2tuesday'")
        if "@weekly_time" in constraints and "3" in constraints["@weekly_time"]: # weekend
            assert poi_num == 2
            sql_constraint.append(f"c{1+max_min_num}.date == '6saturday'")
            sql_constraint.append(f"c{2+max_min_num}.date == '7sunday'")

        sql[-1] = sql[-1][:-1]
        sql = sql + sql_content + ["ON ("] + [" and ".join(sql_constraint)] + [")"]
    elif poi_num > 1 and "max_min_week" in constraints:
        sql_content = ["FROM"]
        sql_constraint = []
        max_min_num = 2 if "@temperature_high_1" in constraints and "@temperature_low_1" in constraints else 1

        sql.append("c1."+sql_cols[0]+",")
        max_min_cursor = 1

        MAXi, MINi = 0, 0
        for i in range(1, max_min_num+1):
            inner_content = ["SELECT DISTINCT"]
            for item in sql_cols:
                if item == "high" and "@temperature_high_1" in constraints and MAXi == 0 and MINi != i:
                    inner_content.append(f"MAX(high) AS Mhigh,")
                    MAXi = i
                elif item == "low" and "@temperature_low_1" in constraints and MAXi != i and MINi == 0:
                    inner_content.append(f"MIN(low) AS Mlow,")
                    MINi= i
                elif item == "location" or item == "date":
                    inner_content.append(f"{item},")
            inner_content[-1] = inner_content[-1][:-1]
            inner_content.append("FROM content GROUP BY location")
            sql_content.append(f"({' '.join(inner_content)}) AS c{i} JOIN")

        for i in range(1, poi_num+1):
            for item in sql_cols[1:]:
                if item == "high" and "@temperature_high_1" in constraints and i == 1 and max_min_cursor <= max_min_num:
                    sql.append(f"c{max_min_cursor}.Mhigh,")
                    max_min_cursor += 1
                elif item == "low" and "@temperature_low_1" in constraints and i == 1 and max_min_cursor <= max_min_num:
                    sql.append(f"c{max_min_cursor}.Mlow,")
                    max_min_cursor += 1
                else:
                    sql.append(f"c{i+max_min_num}.{item},")

            sql_constraint.append(f"c{1+max_min_num}.date == c1.date")
            if i > 1:
                sql_constraint.append(f"c{i}.location == c{i-1}.location")
                sql_constraint.append(f"c{i+max_min_num}.date != c{1+max_min_num}.date")
                if i > 2:
                    sql_constraint.append(f"c{i+max_min_num}.date > c{i+max_min_num-1}.date")
            sql_constraint.append(f"c{i+max_min_num}.location == c{i+max_min_num-1}.location")
            sql_content.append(f"content AS c{i+max_min_num}")
            if i != poi_num:
                sql_content.append("JOIN")

        sql[-1] = sql[-1][:-1]
        sql = sql + sql_content + ["ON ("] + [" and ".join(sql_constraint)] + [")"]
    else: # poi_num > 1 and no more MAX/MIN constraints and no negation
        sql.append("c1."+sql_cols[0]+",")
        sql_content = ["FROM"]
        sql_constraint  = []
        for i in range(1, poi_num+1):
            for item in sql_cols[1:]:
                sql.append(f"c{i}."+item+",")

            if i < poi_num:
                sql_constraint.append(f"c{i+1}.date > c{i}.date")
                sql_constraint.append(f"c{i+1}.location == c{i}.location")
            sql_content.append(f"content AS c{i}")
            if i != poi_num:
                sql_content.append("JOIN")
        if "@weekly_time" in constraints and "1" in constraints["@weekly_time"]: # today
            sql_constraint.append(f"c{1}.date == '1monday'")
        if "@weekly_time" in constraints and "2" in constraints["@weekly_time"]: # tmr
            assert poi_num == 2
            sql_constraint.append(f"c{2}.date == '2tuesday'")
        if "@weekly_time" in constraints and "3" in constraints["@weekly_time"]: # weekend
            assert poi_num == 2
            sql_constraint.append(f"c{1}.date == '6saturday'")
            sql_constraint.append(f"c{2}.date == '7sunday'")

        sql[-1] = sql[-1][:-1] # remove ","
        sql = sql + sql_content + ["ON ("] + [" and ".join(sql_constraint)] + [")"]
    # print("SQL", " ".join(sql))
    # print("\n")
    return " ".join(sql), sql_cols

def generate_schedule_sql(key_group, constraints):
    poi_num = 1
    sql_cols = ["event"]
    for key in key_group:
        poi_num = max([int(max(key_group[key])), poi_num])
        key = key.replace("@", "")
        if key != "event":
            sql_cols.append(key)
    
    sql = ["SELECT DISTINCT"]
    sql.append("c1."+sql_cols[0]+",")
    sql_content = [f"FROM (SELECT DISTINCT event, date, time, room, agenda, party FROM content GROUP BY event HAVING COUNT(event) == {poi_num}) as c1"]
    sql_constraint  = []
    for i in range(1, poi_num+1):
        for item in sql_cols[1:]:
            sql.append(f"c{i}.{item},")
            sql_constraint.append(f"c{i}.{item} != '-'")

        if i < poi_num:
            sql_constraint.append(f"c{i+1}.event == c{i}.event")
        if i > 1:
            sql_content.append(f"content AS c{i}")

            sql_constraint.append(f"( c{i}.date != c1.date or c{i}.time != c1.time )")
            if i < poi_num:
                sql_constraint.append(f"( c{i}.date < c{i+1}.date or c{i}.time < c{i+1}.time )")
        if i != poi_num:
            sql_content.append("JOIN")
    
    sql[-1] = sql[-1][:-1] # remove ","
    prefix = ["WHERE ("]  if poi_num == 1 else ["ON ("] 
    sql = sql + sql_content + prefix + [" and ".join(sql_constraint)] + [")"]
    return " ".join(sql), sql_cols


def generate_navigate_sql(key_group, constraints):
    poi_num = 1
    sql_cols = ["poi_type"]
    for key in key_group:
        poi_num = max([int(max(key_group[key])), poi_num])
        key = key.replace("poi_address", "address").replace("@", "")
        if key != "poi_type":
            sql_cols.append(key)

    sql = ["SELECT DISTINCT"]
    if poi_num == 1 and "max_min" not in constraints:
        """
        c.execute("SELECT DISTINCT poi_type, poi, distance FROM content")
        c.execute("SELECT DISTINCT poi_type, poi, distance, traffic_info FROM content where (traffic_info != 4 and traffic_info != 3 and traffic_info != 2)")
        """
        sql.append(", ".join(sql_cols))
        sql.append("FROM content")
        sql_constraint = []
        if "@traffic_info_1" in constraints and -1 in constraints["@traffic_info_1"]: # avoid heavy traffic
            sql_constraint.append("( traffic_info != 4 and traffic_info != 3 and traffic_info != 2 )")
        if len(sql_constraint) > 0:
            sql.append("WHERE "+" and ".join(sql_constraint))
        return " ".join(sql), sql_cols
    elif poi_num == 1 and "max_min" in constraints:
        """
        c.execute("SELECT DISTINCT poi_type, poi, MIN(distance), address FROM content GROUP BY poi_type")
        c.execute("SELECT DISTINCT poi_type, poi, distance, address, MIN(traffic_info) FROM content GROUP BY poi_type")
        """
        sql_constraint = []
        if "@traffic_info_1" in constraints and -1 in constraints["@traffic_info_1"]: # avoid heavy traffic
            sql_constraint.append("WHERE ( traffic_info != 4 and traffic_info != 3 and traffic_info != 2 )")

        cols = []
        for col in sql_cols:
            if "@traffic_info_1" in constraints and 0 in constraints["@traffic_info_1"] and col == "traffic_info" and "@distance_1" not in constraints:
                cols.append(f"MIN(traffic_info)")
            elif "@distance_1" in constraints and 0 in constraints["@distance_1"] and col == "distance":
                cols.append(f"MIN(distance)")
            else:
                cols.append(col)
        sql.append(", ".join(cols))
        sql.append(f"FROM content {' '.join(sql_constraint)}")
        sql.append("GROUP BY poi_type")
        return " ".join(sql), sql_cols

    if poi_num > 1:
        """
        c.execute("SELECT DISTINCT c1.poi_type, c1.poi, c1.distance, c1.address, c1.traffic_info, c2.poi, c2.distance, c2.address, c2.traffic_info FROM content AS c1 JOIN content AS c2 where ( c1.poi_type == c2.poi_type and c1.traffic_info <= c2.traffic_info and c1.poi != c2.poi)")
        c.execute("SELECT DISTINCT c1.poi_type, c1.poi, c1.distance, c1.address, c1.traffic_info, c2.poi, c2.distance, c2.address, c2.traffic_info FROM content AS c1 JOIN content AS c2 where ( c1.poi_type == c2.poi_type and c1.distance <= c2.distance and c1.poi != c2.poi)")
        """
        sql.append("c1."+sql_cols[0]+",")
        sql_content = ["FROM"]
        sql_constraint  = []
        for i in range(1, poi_num+1):
            for item in sql_cols[1:]:
                sql.append(f"c{i}."+item+",")

            if i < poi_num:
                if "@traffic_info_1" in constraints and 0 in constraints["@traffic_info_1"]:
                    sql_constraint.append(f"c{i+1}.traffic_info > c{i}.traffic_info")
                else:
                    sql_constraint.append(f"c{i+1}.distance > c{i}.distance")
                sql_constraint.append(f"c{i+1}.poi_type == c{i}.poi_type")
            sql_content.append(f"content AS c{i}")
            if i != poi_num:
                sql_content.append("JOIN")
        
        if "@traffic_info_1" in constraints and -1 in constraints["@traffic_info_1"]: # avoid heavy traffic
            sql_constraint.append("c1.traffic_info != 4 and c1.traffic_info != 3 and c1.traffic_info != 2")

        sql[-1] = sql[-1][:-1] # remove ","
        sql = sql + sql_content + ["ON ("] + [" and ".join(sql_constraint)] + [")"]

    return " ".join(sql), sql_cols