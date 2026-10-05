        event_rows.append(
            {
                "Window": row["window"],
                "Diagnosis": row.get("diagnosis"),
                "Alarm": bool(row.get("alarm", False)),
                "Recovery": bool(row.get("recovery", False)),
                "Sensors moved": ", ".join(
                    row.get("features_moved") or []
                ),
                "Worst KS": row.get("worst_ks"),
                "Residual": row.get("mean_resid"),
                **(
                    {"Answer key": row.get("truth_regime")}
                    if show_truth
                    else {}
                ),
            }
        )

    log = pd.DataFrame(event_rows)

    if not log.empty:
        log_slot.dataframe(
            log.sort_values("Window", ascending=False).head(250),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# STREAM EXECUTION
# ============================================================

if start:
    df, pretrained = load_assets()

    runner = DualRunner(
        df,
        TRAIN_END,
        WINDOW,
        pretrained,
    )

    windows = list(
        make_windows(
            len(df),
            TRAIN_END,
            WINDOW,
        )
    )

    previous_truth = None

    progress = st.progress(
        0,
        text="Starting live stream…",
    )

    for i, (s, e) in enumerate(windows):
        msg = runner.step(s, e)

        # Ground truth is kept ONLY as an offline/demo answer key.
        truth = (
            str(df.regime.iloc[s])
            if "regime" in df.columns
            else None
        )

        msg["truth_regime"] = truth

        st.session_state.history.append(msg)

        # True regime transition — visualization/evaluation only.
        if (
            truth is not None
            and previous_truth is not None
            and truth != previous_truth
        ):
            st.session_state.events.append(
                {
                    "window": msg["window"],
                    "type": "truth",
                    "description": f"{previous_truth} → {truth}",
                }
            )

        previous_truth = truth

        # Existing detector alarm.
        if msg.get("alarm"):
            st.session_state.events.append(
                {
                    "window": msg["window"],
                    "type": "alarm",
                    "description": str(
                        msg.get("diagnosis", "")
                    ),
                }
            )

        # Existing recovery/re-adaptation event.
        if msg.get("recovery"):
            st.session_state.events.append(
                {
                    "window": msg["window"],
                    "type": "adapt",
                    "description": "Drift ended → re-adapt",
                }
            )

        # Redraw every few windows rather than on every observation.
        if i % 3 == 0 or i == len(windows) - 1:
            render_dashboard()

            progress.progress(
                (i + 1) / len(windows),
                text=f"Streaming window {i + 1}/{len(windows)}",
            )

        time.sleep(delay)

    progress.empty()
    st.success("Stream finished.")

elif st.session_state.history:
    render_dashboard()

else:
