def test_active_dagster_graph_contains_only_market_ingestion():
    from lab.platform.dagster import defs

    asset_names = {
        key.to_user_string()
        for asset_definition in defs.assets
        for key in asset_definition.keys
    }
    check_names = {spec.name for check in defs.asset_checks for spec in check.check_specs}
    schedule_names = {schedule.name for schedule in defs.schedules}

    assert asset_names == {"ingestion_publication"}
    assert check_names == {"ingestion_calendar_coverage"}
    assert schedule_names == {"daily_market_data_ingestion", "hourly_market_data_ingestion"}
