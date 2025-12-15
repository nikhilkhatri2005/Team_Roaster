import numpy as np
import pandas as pd
import plotly.express as px
from math import ceil
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="Roster Analysis", layout="wide")

# ---------------- Constants ----------------
FILE_PATH = "New_roster_tidy_clean.csv"

LEADS = {
    "Nikhil Ghatge",
    "Siddharth Gujral",
    "Namrata Agrawal",
    "Shalaka Jaitapkar",
    "Santosh Kumar Kumsi",
}

BASE_SHIFT_MAP = {
    "First Shift": "First Shift",
    "Second Shift": "Second Shift",
    "General Shift": "Second Shift",
    "Third Shift": "Third Shift",
    "Weekly Off": "Off",
    "Holiday": "Holiday",
    "Planned Leave": "Leave",
    "Half Day": "Leave",
    "Comp Off": "Leave",
    "Training": "Training",
}
LEAVE_DETAILS = {"Planned Leave", "Half Day", "Comp Off"}
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WORKING_SHIFT_ORDER = ["First Shift", "Second Shift", "Third Shift"]
ALL_SHIFT_CATEGORIES = ["First Shift", "Second Shift", "Third Shift", "Off", "Holiday", "Leave", "Training", "Other"]
LEAVE_ORDER = ["Planned Leave", "Half Day", "Comp Off"]

# Labels for cleaner axes/legends
LABELS = {
    "MonthLabel": "Month",
    "MonthStart": "Month",
    "person_days": "Person-days",
    "unique_people": "Unique people",
    "daily_headcount": "Avg daily headcount",
    "avg_daily_headcount": "Avg daily headcount",
    "headcount": "Headcount",
    "ISOWeekLabel": "Week",
    "Shift_Core": "Shift",
    "Shift Detail": "Leave type",
    "Resource Name": "Person",
    "Team_Analyzed": "Team",
    "working_days": "Working days",
    "ThirdShiftShare": "Third-shift share (%)",
}

# ---------------- Utilities ----------------
def _facet_height(n_panels: int, per_row: int = 5, base: int = 170, min_height: int = 600) -> int:
    rows = ceil(n_panels / per_row) if n_panels else 1
    return max(min_height, rows * base)

def add_data_labels(fig, chart_type="bar", horizontal=False, decimals=0, stacked=False, show_labels=True):
    if not show_labels:
        return fig
    if chart_type == "bar":
        if stacked:
            fig.update_traces(
                texttemplate=f"%{{y:.{decimals}f}}",
                textposition="inside",
                insidetextanchor="middle",
                textfont_color="white",
                cliponaxis=False,
            )
        else:
            if horizontal:
                fig.update_traces(
                    texttemplate=f"%{{x:.{decimals}f}}",
                    textposition="outside",
                    cliponaxis=False,
                )
            else:
                fig.update_traces(
                    texttemplate=f"%{{y:.{decimals}f}}",
                    textposition="outside",
                    cliponaxis=False,
                )
    elif chart_type == "line":
        fig.update_traces(
            mode="lines+markers+text",
            texttemplate=f"%{{y:.{decimals}f}}",
            textposition="top center",
        )
    elif chart_type == "heatmap":
        fig.update_traces(texttemplate=f"%{{z:.{decimals}f}}")
    return fig

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    # Dates and periods
    df["Date_dt"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df["MonthStart"] = df["Date_dt"].dt.to_period("M").dt.to_timestamp()
    df["MonthLabel"] = df["MonthStart"].dt.strftime("%b-%y")
    isocal = df["Date_dt"].dt.isocalendar()
    df["ISOYear"] = isocal.year
    df["ISOWeekNum"] = isocal.week.astype(int)
    df["ISOWeekLabel"] = "CW" + df["ISOWeekNum"].astype(str).str.zfill(2)
    df["DOW"] = df["Date_dt"].dt.day_name().str[:3]
    df["IsWeekend"] = df["Date_dt"].dt.dayofweek >= 5
    # Team label (vectorized)
    df["Team_Analyzed"] = np.where(df["Resource Name"].isin(LEADS), "Lead", df["Team Details"])
    return df

def derive_shift_core(df: pd.DataFrame, merge_l15_into_second: bool = True) -> pd.Series:
    shift_map = BASE_SHIFT_MAP.copy()
    shift_map["L1.5 Timings"] = "Second Shift" if merge_l15_into_second else "L1.5 Timings"
    return df["Shift Detail"].map(shift_map).fillna("Other")

def apply_filters(
    df: pd.DataFrame,
    date_range,
    teams,
    include_leads: bool,
    people,
    include_weekends: bool,
    include_weekdays: bool,
    allowed_shifts,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    # Date range
    if date_range:
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2 and all(date_range):
            start, end = date_range
        else:
            start = end = date_range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask &= df["Date_dt"].between(start_ts, end_ts)

    # Teams and leads
    if teams:
        team_mask = df["Team_Analyzed"].isin(teams)
    else:
        team_mask = pd.Series(False, index=df.index)
    if include_leads:
        team_mask |= df["Team_Analyzed"].eq("Lead")
    if not teams and not include_leads:
        team_mask = True
    mask &= team_mask

    # People
    if people:
        mask &= df["Resource Name"].isin(people)

    # Weekend/weekday
    dow = df["Date_dt"].dt.dayofweek
    is_weekend = dow >= 5
    ww_mask = (is_weekend & include_weekends) | (~is_weekend & include_weekdays)
    mask &= ww_mask

    # Shift category
    if allowed_shifts:
        mask &= df["Shift_Core"].isin(allowed_shifts)

    return df[mask].copy()

def month_order_from_df(df: pd.DataFrame) -> list[str]:
    return df["MonthStart"].dropna().sort_values().drop_duplicates().dt.strftime("%b-%y").tolist()

def week_label_order_from_df(df: pd.DataFrame) -> list[str]:
    wk = (
        df[["ISOYear", "ISOWeekNum", "ISOWeekLabel"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["ISOYear", "ISOWeekNum"])
    )
    return wk["ISOWeekLabel"].tolist()

def render_help():
    with st.sidebar.expander("Help", expanded=False):
        st.write(
            "- Use filters to restrict dates, teams, people, weekdays/weekends, and shift categories.\n"
            "- Merge 'L1.5 Timings' includes L1.5 under Second Shift.\n"
            "- Show chart value labels toggles data labels on bars/lines.\n"
            "- Smooth lines (spline) makes monthly line charts less jagged.\n"
            "\n"
            "Tabs:\n"
            "1) Monthly Shift Mix: Monthly person-days, unique staffing, avg daily headcount by shift.\n"
            "2) Weekly & Leaderboards: Weekly unique headcount + top weekend, night, and leave contributors.\n"
            "3) Calendar Heatmap: Avg headcount by Month vs Day-of-Week, plus weekday vs weekend trend.\n"
            "4) Team → Person → Shift: Sunburst and trends by team and third-shift share.\n"
            "5) Individual Analyzer: One person’s monthly working days, shift mix, leave mix, and KPIs.\n"
            "6) People Comparison: Compare monthly working days and shift mix for selected people."
        )

# ---------------- Tab renderers ----------------
def render_tab1_monthly_mix(filtered: pd.DataFrame, month_order: list[str], show_labels: bool, smooth_lines: bool):
    st.subheader("Monthly Shift Mix")
    if filtered.empty:
        st.info("No data for current filters.")
        return

    working = filtered[filtered["IsWorking"]]

    # Stacked bars: monthly person-days by shift
    monthly_pd = (
        working.groupby(["MonthLabel", "Shift_Core"])
        .size()
        .reset_index(name="person_days")
    )
    if monthly_pd.empty:
        st.info("No working person-days for current filters.")
    else:
        fig1 = px.bar(
            monthly_pd,
            x="MonthLabel",
            y="person_days",
            color="Shift_Core",
            labels=LABELS,
            title="Person-days by Month and Shift",
            barmode="stack",
            category_orders={"MonthLabel": month_order, "Shift_Core": WORKING_SHIFT_ORDER},
        )
        fig1.update_layout(legend_title_text="Shift")
        fig1 = add_data_labels(fig1, chart_type="bar", stacked=True, show_labels=show_labels)
        st.plotly_chart(fig1, use_container_width=True)

    # Grouped bars: unique people by shift
    monthly_unique = (
        working.groupby(["MonthLabel", "Shift_Core"])["Resource Name"]
        .nunique()
        .reset_index(name="unique_people")
    )
    if not monthly_unique.empty:
        fig2 = px.bar(
            monthly_unique,
            x="MonthLabel",
            y="unique_people",
            color="Shift_Core",
            labels=LABELS,
            title="Staffing Count (Unique People) by Month and Shift",
            barmode="group",
            category_orders={"MonthLabel": month_order, "Shift_Core": WORKING_SHIFT_ORDER},
        )
        fig2.update_layout(legend_title_text="Shift")
        fig2 = add_data_labels(fig2, chart_type="bar", show_labels=show_labels)
        st.plotly_chart(fig2, use_container_width=True)

    # Line: avg daily headcount by shift (use MonthStart on x)
    if not working.empty:
        daily_headcount = (
            working.groupby(["Date_dt", "Shift_Core"])["Resource Name"]
            .nunique()
            .reset_index(name="daily_headcount")
        )
        daily_headcount["MonthStart"] = daily_headcount["Date_dt"].dt.to_period("M").dt.to_timestamp()
        avg_monthly_headcount = (
            daily_headcount.groupby(["MonthStart", "Shift_Core"])["daily_headcount"]
            .mean()
            .reset_index(name="avg_daily_headcount")
            .sort_values(["Shift_Core", "MonthStart"])
        )
        fig3 = px.line(
            avg_monthly_headcount,
            x="MonthStart",
            y="avg_daily_headcount",
            color="Shift_Core",
            labels=LABELS,
            title="Average Daily Headcount per Month (by Shift)",
        )
        if smooth_lines:
            fig3.update_traces(line_shape="spline")
        fig3.update_traces(connectgaps=False, mode="lines+markers")
        fig3.update_layout(legend_title_text="Shift", xaxis=dict(tickformat="%b-%y"))
        st.plotly_chart(fig3, use_container_width=True)

def render_tab2_weekly_leaderboards(filtered: pd.DataFrame, week_label_order: list[str], show_labels: bool):
    st.subheader("Weekly Unique Headcount and Leaderboards")
    if filtered.empty:
        st.info("No data for current filters.")
        return

    working = filtered[filtered["IsWorking"]]
    weekly_unique = (
        working.groupby(["ISOYear", "ISOWeekNum", "ISOWeekLabel"])["Resource Name"]
        .nunique()
        .reset_index(name="unique_people")
        .sort_values(["ISOYear", "ISOWeekNum"])
    )
    if weekly_unique.empty:
        st.info("No weekly headcount available.")
    else:
        figWk = px.bar(
            weekly_unique,
            x="ISOWeekLabel",
            y="unique_people",
            labels={"unique_people": "Unique people", "ISOWeekLabel": "Week"},
            title="Weekly Unique Headcount (Worked ≥1 day)",
            category_orders={"ISOWeekLabel": week_label_order},
        )
        figWk = add_data_labels(figWk, chart_type="bar", show_labels=show_labels)
        st.plotly_chart(figWk, use_container_width=True)

    colA, colB, colC = st.columns(3)

    weekend_work = working[working["IsWeekend"]]
    if not weekend_work.empty:
        weekend_lb = (
            weekend_work.groupby("Resource Name")
            .size()
            .sort_values(ascending=False)
            .head(15)
        )
        figWW = px.bar(
            weekend_lb[::-1],
            orientation="h",
            labels={"value": "Person-days", "index": "Person"},
            title="Top Weekend Contributors (Person-days)",
        )
        figWW = add_data_labels(figWW, chart_type="bar", horizontal=True, show_labels=show_labels)
        colA.plotly_chart(figWW, use_container_width=True)

    night = working[working["Shift_Core"] == "Third Shift"]
    if not night.empty:
        night_lb = (
            night.groupby("Resource Name")
            .size()
            .sort_values(ascending=False)
            .head(15)
        )
        figN = px.bar(
            night_lb[::-1],
            orientation="h",
            labels={"value": "Third-shift days", "index": "Person"},
            title="Top Night Shift (Third) Contributors",
        )
        figN = add_data_labels(figN, chart_type="bar", horizontal=True, show_labels=show_labels)
        colB.plotly_chart(figN, use_container_width=True)

    leave = filtered[filtered["Shift_Core"] == "Leave"]
    if not leave.empty:
        leave_lb = (
            leave.groupby("Resource Name")
            .size()
            .sort_values(ascending=False)
            .head(15)
        )
        figL = px.bar(
            leave_lb[::-1],
            orientation="h",
            labels={"value": "Leave days", "index": "Person"},
            title="Leave Utilization by Person (Person-days)",
        )
        figL = add_data_labels(figL, chart_type="bar", horizontal=True, show_labels=show_labels)
        colC.plotly_chart(figL, use_container_width=True)

def render_tab3_calendar_heatmap(filtered: pd.DataFrame, month_order: list[str], show_labels: bool, smooth_lines: bool):
    st.subheader("Calendar Heatmap (Month vs Day-of-Week)")
    if filtered.empty:
        st.info("No data for current filters.")
        return

    shifts_for_heatmap = st.multiselect(
        "Shifts to include in heatmap",
        ALL_SHIFT_CATEGORIES,
        default=WORKING_SHIFT_ORDER,
    )
    heat_df = filtered[filtered["Shift_Core"].isin(shifts_for_heatmap)].copy()
    if heat_df.empty:
        st.info("No rows for the selected shift set.")
        return

    daily_headcount = (
        heat_df.groupby(["Date_dt"])["Resource Name"]
        .nunique()
        .reset_index(name="headcount")
    )
    daily_headcount["MonthLabel"] = daily_headcount["Date_dt"].dt.strftime("%b-%y")
    daily_headcount["DOW"] = daily_headcount["Date_dt"].dt.day_name().str[:3]
    avg_by_month_dow = (
        daily_headcount.groupby(["MonthLabel", "DOW"])["headcount"]
        .mean()
        .reset_index()
    )
    avg_by_month_dow["DOW"] = pd.Categorical(avg_by_month_dow["DOW"], categories=DOW_ORDER, ordered=True)
    avg_by_month_dow["MonthLabel"] = pd.Categorical(avg_by_month_dow["MonthLabel"], categories=month_order, ordered=True)

    figH = px.density_heatmap(
        avg_by_month_dow,
        x="MonthLabel",
        y="DOW",
        z="headcount",
        labels=LABELS,
        title="Average Daily Headcount by Month and Day-of-Week",
        category_orders={"MonthLabel": month_order, "DOW": DOW_ORDER},
        color_continuous_scale="Blues",
    )
    figH = add_data_labels(figH, chart_type="heatmap", decimals=1, show_labels=show_labels)
    st.plotly_chart(figH, use_container_width=True)

    # Trend: use MonthStart on x to avoid zig-zag
    daily_headcount["IsWeekend"] = daily_headcount["Date_dt"].dt.dayofweek >= 5
    daily_headcount["MonthStart"] = daily_headcount["Date_dt"].dt.to_period("M").dt.to_timestamp()
    ww_month = (
        daily_headcount.groupby(["MonthStart", "IsWeekend"])["headcount"]
        .mean()
        .reset_index()
        .sort_values(["IsWeekend", "MonthStart"])
    )
    ww_month["DayType"] = ww_month["IsWeekend"].map({True: "Weekend", False: "Weekday"})

    figWW = px.line(
        ww_month,
        x="MonthStart",
        y="headcount",
        color="DayType",
        labels={"MonthStart": "Month", "headcount": "Headcount", "DayType": "Day type"},
        title="Average Headcount Trend: Weekday vs Weekend",
    )
    if smooth_lines:
        figWW.update_traces(line_shape="spline")
    figWW.update_traces(connectgaps=False, mode="lines+markers")
    figWW.update_layout(xaxis=dict(tickformat="%b-%y"))
    st.plotly_chart(figWW, use_container_width=True)

def render_tab4_team_person_shift(filtered: pd.DataFrame, month_order: list[str], show_labels: bool, smooth_lines: bool):
    st.subheader("Team → Person → Shift Sunburst and Trends")
    if filtered.empty:
        st.info("No data for current filters.")
        return

    # Sunburst
    sb = (
        filtered.groupby(["Team_Analyzed", "Resource Name", "Shift_Core"])
        .size()
        .reset_index(name="person_days")
    )
    if sb.empty:
        st.info("Not enough data for sunburst.")
    else:
        figS = px.sunburst(
            sb,
            path=["Team_Analyzed", "Resource Name", "Shift_Core"],
            values="person_days",
            title="Team → Person → Shift Sunburst",
            labels={"Team_Analyzed": "Team", "Resource Name": "Person", "Shift_Core": "Shift", "person_days": "Person-days"},
        )
        figS.update_layout(height=800)
        st.plotly_chart(figS, use_container_width=True)

    # Trends
    working = filtered[filtered["IsWorking"]].copy()
    if working.empty:
        return

    # Third-shift share by month (MonthStart)
    monthly_counts = (
        working.groupby("MonthStart")
        .size()
        .reset_index(name="total")
        .sort_values("MonthStart")
    )
    monthly_third = (
        working[working["Shift_Core"] == "Third Shift"]
        .groupby("MonthStart")
        .size()
        .reset_index(name="third")
        .sort_values("MonthStart")
    )
    third_share = pd.merge(monthly_counts, monthly_third, on="MonthStart", how="left").fillna(0)
    third_share["ThirdShiftShare"] = (third_share["third"] / third_share["total"] * 100).round(2)

    figShare = px.line(
        third_share,
        x="MonthStart",
        y="ThirdShiftShare",
        markers=True,
        labels={"MonthStart": "Month", "ThirdShiftShare": "Third-shift share (%)"},
        title="Monthly Third-Shift Share (%)",
    )
    if smooth_lines:
        figShare.update_traces(line_shape="spline")
    figShare.update_traces(connectgaps=False, mode="lines+markers")
    figShare.update_layout(xaxis=dict(tickformat="%b-%y"))
    st.plotly_chart(figShare, use_container_width=True)

    # Monthly person-days by Team (MonthStart)
    team_month = (
        working.groupby(["MonthStart", "Team_Analyzed"])
        .size()
        .reset_index(name="person_days")
        .sort_values(["Team_Analyzed", "MonthStart"])
    )
    figTeam = px.line(
        team_month,
        x="MonthStart",
        y="person_days",
        color="Team_Analyzed",
        markers=True,
        labels={"MonthStart": "Month", "person_days": "Person-days", "Team_Analyzed": "Team"},
        title="Monthly Person-days by Team (Working Shifts)",
    )
    if smooth_lines:
        figTeam.update_traces(line_shape="spline")
    figTeam.update_traces(connectgaps=False, mode="lines+markers")
    figTeam.update_layout(xaxis=dict(tickformat="%b-%y"))
    st.plotly_chart(figTeam, use_container_width=True)

def render_tab5_individual_analyzer(filtered: pd.DataFrame, month_order: list[str], show_labels: bool, smooth_lines: bool):
    st.subheader("Individual Analyzer")
    if filtered.empty:
        st.info("No data for current filters.")
        return

    people_in_scope = sorted(filtered["Resource Name"].unique().tolist())
    if not people_in_scope:
        st.info("No people for current filters.")
        return

    person = st.selectbox("Select person", options=people_in_scope, key="ind_person")
    person_df = filtered[filtered["Resource Name"] == person].copy()
    if person_df.empty:
        st.info("No data for selected person.")
        return

    # KPIs
    working_p = person_df[person_df["IsWorking"]]
    leave_p = person_df[person_df["Shift Detail"].isin(LEAVE_DETAILS)]
    weekend_work_p = working_p[working_p["IsWeekend"]]
    nights_p = working_p[working_p["Shift_Core"] == "Third Shift"]

    total_work_days = int(len(working_p))
    total_leave_days = int(len(leave_p))
    weekend_days = int(len(weekend_work_p))
    night_days = int(len(nights_p))
    night_share = round((night_days / total_work_days * 100), 1) if total_work_days else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Working days", total_work_days)
    k2.metric("Leave days", total_leave_days)
    k3.metric("Weekend working days", weekend_days)
    k4.metric("Night (Third) days", f"{night_days} ({night_share}%)")

    # Monthly working days line (MonthStart)
    per_month = (
        working_p.groupby("MonthStart")
        .size()
        .reset_index(name="working_days")
        .sort_values("MonthStart")
    )
    fig1 = px.line(
        per_month,
        x="MonthStart",
        y="working_days",
        markers=True,
        labels={"MonthStart": "Month", "working_days": "Working days"},
        title=f"Monthly Working Days — {person}",
    )
    if smooth_lines:
        fig1.update_traces(line_shape="spline")
    fig1.update_traces(connectgaps=False, mode="lines+markers")
    fig1.update_layout(xaxis=dict(tickformat="%b-%y"))
    st.plotly_chart(fig1, use_container_width=True)

    # Monthly shift mix (stacked)
    per_shift = (
        working_p.groupby(["MonthLabel", "Shift_Core"])
        .size()
        .reset_index(name="person_days")
    )
    if not per_shift.empty:
        per_shift["MonthLabel"] = pd.Categorical(per_shift["MonthLabel"], categories=month_order, ordered=True)
        per_shift["Shift_Core"] = pd.Categorical(per_shift["Shift_Core"], categories=WORKING_SHIFT_ORDER, ordered=True)
        fig2 = px.bar(
            per_shift,
            x="MonthLabel",
            y="person_days",
            color="Shift_Core",
            barmode="stack",
            labels={"MonthLabel": "Month", "person_days": "Person-days", "Shift_Core": "Shift"},
            title=f"Monthly Shift Mix — {person}",
            category_orders={"MonthLabel": month_order, "Shift_Core": WORKING_SHIFT_ORDER},
        )
        fig2.update_layout(legend_title_text="Shift")
        fig2 = add_data_labels(fig2, chart_type="bar", stacked=True, show_labels=show_labels)
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly leave by type
    per_leave = (
        person_df[person_df["Shift Detail"].isin(LEAVE_DETAILS)]
        .groupby(["MonthLabel", "Shift Detail"])
        .size()
        .reset_index(name="person_days")
    )
    if not per_leave.empty:
        per_leave["MonthLabel"] = pd.Categorical(per_leave["MonthLabel"], categories=month_order, ordered=True)
        per_leave["Shift Detail"] = pd.Categorical(per_leave["Shift Detail"], categories=LEAVE_ORDER, ordered=True)
        fig3 = px.bar(
            per_leave,
            x="MonthLabel",
            y="person_days",
            color="Shift Detail",
            barmode="group",
            labels={"MonthLabel": "Month", "person_days": "Person-days", "Shift Detail": "Leave type"},
            title=f"Monthly Leave by Type — {person}",
            category_orders={"MonthLabel": month_order, "Shift Detail": LEAVE_ORDER},
        )
        fig3.update_layout(legend_title_text="Leave type")
        fig3 = add_data_labels(fig3, chart_type="bar", show_labels=show_labels)
        st.plotly_chart(fig3, use_container_width=True)

def render_tab6_people_comparison(filtered: pd.DataFrame, month_order: list[str], show_labels: bool, smooth_lines: bool):
    st.subheader("People Comparison")
    if filtered.empty:
        st.info("No data for current filters.")
        return

    working = filtered[filtered["IsWorking"]]
    if working.empty:
        st.info("No working shifts for current filters.")
        return

    # Default: top 5 by working days
    work_counts = working.groupby("Resource Name").size().sort_values(ascending=False)
    default_people = work_counts.head(5).index.tolist()
    people_pool = sorted(working["Resource Name"].unique().tolist())
    sel_people = st.multiselect("Select people to compare", people_pool, default=default_people, key="compare_people")

    if not sel_people:
        st.info("Select at least one person to compare.")
        return

    comp = working[working["Resource Name"].isin(sel_people)].copy()

    # Monthly working days by person (MonthStart)
    per_person_month = (
        comp.groupby(["Resource Name", "MonthStart"])
        .size()
        .reset_index(name="working_days")
        .sort_values(["Resource Name", "MonthStart"])
    )
    fig1 = px.line(
        per_person_month,
        x="MonthStart",
        y="working_days",
        color="Resource Name",
        markers=True,
        labels={"MonthStart": "Month", "working_days": "Working days", "Resource Name": "Person"},
        title="Monthly Working Days by Person",
    )
    if smooth_lines:
        fig1.update_traces(line_shape="spline")
    fig1.update_traces(connectgaps=False, mode="lines+markers")
    fig1.update_layout(xaxis=dict(tickformat="%b-%y"))
    st.plotly_chart(fig1, use_container_width=True)

    # Shift mix by person (percent across selected period) - compute percent manually
    per_person_shift = (
        comp.groupby(["Resource Name", "Shift_Core"])
        .size()
        .reset_index(name="person_days")
    )
    per_person_shift = per_person_shift[per_person_shift["Shift_Core"].isin(WORKING_SHIFT_ORDER)]
    if not per_person_shift.empty:
        totals = per_person_shift.groupby("Resource Name")["person_days"].transform("sum")
        per_person_shift["percent"] = per_person_shift["person_days"] / totals * 100.0
        per_person_shift["Shift_Core"] = pd.Categorical(per_person_shift["Shift_Core"], categories=WORKING_SHIFT_ORDER, ordered=True)

        fig2 = px.bar(
            per_person_shift,
            x="Resource Name",
            y="percent",
            color="Shift_Core",
            barmode="stack",
            labels={"Resource Name": "Person", "percent": "Shift mix (%)", "Shift_Core": "Shift"},
            category_orders={"Shift_Core": WORKING_SHIFT_ORDER},
            title="Shift Mix by Person (Percent across selected period)",
        )
        fig2.update_layout(legend_title_text="Shift")
        fig2.update_yaxes(range=[0, 100], ticksuffix="%")
        fig2 = add_data_labels(fig2, chart_type="bar", stacked=True, show_labels=show_labels)
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- Main app ----------------
def main():
    st.title("Roster Analysis Dashboard")
    st.caption("Interactive analysis of New_roster_tidy_clean.csv with tabs")

    with st.sidebar:
        st.header("Filters")
        merge_l15 = st.toggle("Merge 'L1.5 Timings' into Second Shift", value=True)
        show_labels = st.checkbox("Show chart value labels", value=True)
        smooth_lines = st.checkbox("Smooth lines (spline)", value=True)

    df = load_data(FILE_PATH)

    # Shift core and flags
    df["Shift_Core"] = derive_shift_core(df, merge_l15_into_second=merge_l15)
    df["IsWorking"] = df["Shift_Core"].isin(WORKING_SHIFT_ORDER)
    df["IsLeave"] = df["Shift Detail"].isin(LEAVE_DETAILS)

    with st.sidebar:
        # Date range
        min_dt, max_dt = df["Date_dt"].min(), df["Date_dt"].max()
        date_range = st.date_input(
            "Date range",
            value=(min_dt.date(), max_dt.date()),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
        )

        teams_all = sorted([t for t in df["Team_Analyzed"].unique() if t != "Lead"])
        sel_teams = st.multiselect("Teams", teams_all, default=teams_all)
        include_leads = st.checkbox("Include Leads", value=True)

        people_all = sorted(df["Resource Name"].unique())
        sel_people = st.multiselect("People (optional)", people_all)

        include_weekdays = st.checkbox("Include Weekdays", value=True)
        include_weekends = st.checkbox("Include Weekends", value=True)

        allowed_shifts = st.multiselect(
            "Shift categories",
            ALL_SHIFT_CATEGORIES,
            default=["First Shift", "Second Shift", "Third Shift", "Off", "Holiday", "Leave"],
        )

    # Apply filters
    filtered = apply_filters(
        df,
        date_range=date_range,
        teams=sel_teams,
        include_leads=include_leads,
        people=sel_people,
        include_weekends=include_weekends,
        include_weekdays=include_weekdays,
        allowed_shifts=allowed_shifts,
    )

    # Orders
    month_order = month_order_from_df(filtered)
    week_label_order = week_label_order_from_df(filtered)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Monthly Shift Mix",
        "Weekly & Leaderboards",
        "Calendar Heatmap",
        "Team → Person → Shift",
        "Individual Analyzer",
        "People Comparison",
    ])

    with tab1:
        render_tab1_monthly_mix(filtered, month_order, show_labels, smooth_lines)

    with tab2:
        render_tab2_weekly_leaderboards(filtered, week_label_order, show_labels)

    with tab3:
        render_tab3_calendar_heatmap(filtered, month_order, show_labels, smooth_lines)

    with tab4:
        render_tab4_team_person_shift(filtered, month_order, show_labels, smooth_lines)

    with tab5:
        render_tab5_individual_analyzer(filtered, month_order, show_labels, smooth_lines)

    with tab6:
        render_tab6_people_comparison(filtered, month_order, show_labels, smooth_lines)

    # Help
    render_help()

    # Footer preview and export
    st.markdown("### Data preview (filtered)")
    if filtered.empty:
        st.info("No rows to display for current filters.")
    else:
        st.dataframe(
            filtered.sort_values(["Date_dt", "Resource Name"]).drop(columns=["Date_dt"]),
            use_container_width=True,
        )
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="filtered_roster.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
