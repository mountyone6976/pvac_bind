import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr, spearmanr


st.set_page_config(page_title="pVACbind Viewer", layout="wide")


def read_uploaded_tsv(uploaded_file, label):
    try:
        return pd.read_csv(uploaded_file, sep="\t", low_memory=False)
    except Exception as exc:
        st.error(f"Could not read {label} as a tab-separated TSV file: {exc}")
        st.stop()


def require_columns(frame, required_columns, label):
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        st.error(
            f"{label} is missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(frame.columns.astype(str))}"
        )
        st.stop()


def numeric_column(frame, column_name, new_column_name=None):
    if column_name not in frame.columns:
        return None
    target = new_column_name or f"{column_name} numeric"
    frame[target] = pd.to_numeric(frame[column_name], errors="coerce")
    return target


def extract_gene_from_id(value):
    if pd.isna(value):
        return np.nan
    return str(value).split("|")[0]


def add_bar_labels(fig, texttemplate=None):
    kwargs = {"textposition": "outside", "cliponaxis": False}
    if texttemplate:
        kwargs["texttemplate"] = texttemplate
    fig.update_traces(**kwargs)
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")
    return fig


def download_text(label, text, file_name):
    st.download_button(
        label,
        text,
        file_name=file_name,
        mime="text/plain",
    )


def safe_corr(x, y, method_name):
    corr_df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(corr_df) < 2:
        return np.nan
    if corr_df["x"].nunique() < 2 or corr_df["y"].nunique() < 2:
        return np.nan
    if method_name == "pearson":
        return pearsonr(corr_df["x"], corr_df["y"]).statistic
    if method_name == "spearman":
        return spearmanr(corr_df["x"], corr_df["y"]).statistic
    return np.nan


def build_sidebar_filters(frame):
    filtered = frame.copy()

    st.sidebar.header("Filters")

    if "Tier" in filtered.columns:
        tiers = sorted(filtered["Tier"].dropna().astype(str).unique())
        selected_tiers = st.sidebar.multiselect("Tier", tiers, default=tiers)
        if selected_tiers:
            filtered = filtered[filtered["Tier"].astype(str).isin(selected_tiers)]

    if "Allele" in filtered.columns:
        alleles = sorted(filtered["Allele"].dropna().astype(str).unique())
        selected_alleles = st.sidebar.multiselect("Allele", alleles, default=alleles)
        if selected_alleles:
            filtered = filtered[filtered["Allele"].astype(str).isin(selected_alleles)]

    if "IC50 MT" in filtered.columns:
        ic50_col = numeric_column(filtered, "IC50 MT", "IC50 MT numeric")
        max_ic50 = st.sidebar.number_input("Max IC50 MT", value=50000.0, min_value=0.0)
        filtered = filtered[filtered[ic50_col].isna() | (filtered[ic50_col] <= max_ic50)]

    percentile_cols = [col for col in filtered.columns if "%ile MT" in str(col)]
    for col in percentile_cols:
        numeric_col = numeric_column(filtered, col, f"{col} numeric")
        valid_values = filtered[numeric_col].dropna()
        default_value = float(valid_values.max()) if not valid_values.empty else 100.0
        max_percentile = st.sidebar.number_input(
            f"Max {col}",
            value=default_value,
            min_value=0.0,
            key=f"max_{col}",
        )
        filtered = filtered[filtered[numeric_col].isna() | (filtered[numeric_col] <= max_percentile)]

    if "Best Peptide" in filtered.columns:
        peptide_search = st.sidebar.text_input("Search peptide")
        if peptide_search:
            filtered = filtered[
                filtered["Best Peptide"].astype(str).str.contains(peptide_search, case=False, na=False)
            ]

    if "ID" in filtered.columns:
        id_search = st.sidebar.text_input("Search ID / gene")
        if id_search:
            filtered = filtered[
                filtered["ID"].astype(str).str.contains(id_search, case=False, na=False)
            ]

    return filtered


def show_pvacbind_summary(df, filtered):
    st.header("pVACbind Summary")

    total_rows = len(df)
    summary_lines = [
        f"Total rows: {total_rows:,}",
        f"Total columns: {df.shape[1]:,}",
        f"Filtered rows: {len(filtered):,}",
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows", f"{total_rows:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Filtered rows", f"{len(filtered):,}")

    if "Tier" in df.columns:
        pass_count = (df["Tier"].astype(str) == "Pass").sum()
        pass_pct = pass_count / total_rows * 100 if total_rows else 0
        col4.metric("Pass rows", f"{pass_count:,}", f"{pass_pct:.2f}%")
        summary_lines.append(f"Pass rows: {pass_count:,}")
        summary_lines.append(f"Percent Pass: {pass_pct:.2f}%")

    if "Best Peptide" in df.columns:
        summary_lines.append(f"Unique Best Peptide values: {df['Best Peptide'].nunique():,}")

    if "Allele" in df.columns:
        summary_lines.append(f"Unique alleles: {df['Allele'].nunique():,}")

    if "Best Peptide" in df.columns and "Allele" in df.columns:
        unique_pairs = df[["Best Peptide", "Allele"]].drop_duplicates().shape[0]
        summary_lines.append(f"Unique Best Peptide + Allele pairs: {unique_pairs:,}")

        if "Tier" in df.columns:
            pass_unique_pairs = (
                df[df["Tier"].astype(str) == "Pass"][["Best Peptide", "Allele"]]
                .drop_duplicates()
                .shape[0]
            )
            summary_lines.append(f"Unique passing Best Peptide + Allele pairs: {pass_unique_pairs:,}")

    if "IC50 MT" in df.columns:
        ic50 = pd.to_numeric(df["IC50 MT"], errors="coerce")
        summary_lines.append(f"Median IC50 MT: {ic50.median():.3f}")
        summary_lines.append(f"Mean IC50 MT: {ic50.mean():.3f}")
        summary_lines.append(f"Minimum IC50 MT: {ic50.min():.3f}")
        summary_lines.append(f"Rows with IC50 MT <= 500: {(ic50 <= 500).sum():,}")
        summary_lines.append(f"Rows with IC50 MT <= 50: {(ic50 <= 50).sum():,}")

    if "%ile MT" in df.columns:
        percentile = pd.to_numeric(df["%ile MT"], errors="coerce")
        summary_lines.append(f"Median %ile MT: {percentile.median():.3f}")
        summary_lines.append(f"Rows with %ile MT <= 2: {(percentile <= 2).sum():,}")
        summary_lines.append(f"Rows with %ile MT <= 0.5: {(percentile <= 0.5).sum():,}")

    if "Pres %ile MT" in df.columns:
        pres = pd.to_numeric(df["Pres %ile MT"], errors="coerce")
        summary_lines.append(f"Median Pres %ile MT: {pres.median():.3f}")
        summary_lines.append(f"Rows with Pres %ile MT <= 2: {(pres <= 2).sum():,}")
        summary_lines.append(f"Rows with Pres %ile MT <= 0.5: {(pres <= 0.5).sum():,}")

    st.subheader("Filtered table")
    st.write(f"Filtered rows: {filtered.shape[0]:,}")
    st.dataframe(filtered, use_container_width=True, height=600)

    st.download_button(
        "Download filtered TSV",
        filtered.to_csv(sep="\t", index=False),
        file_name="filtered_pvacbind.tsv",
        mime="text/tab-separated-values",
    )

    st.header("Charts")

    if "Tier" in df.columns:
        tier_counts = df["Tier"].astype(str).value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig = px.bar(tier_counts, x="Tier", y="Count", text="Count", title="Tier counts")
        st.plotly_chart(add_bar_labels(fig), use_container_width=True)

    if "Allele" in df.columns:
        allele_counts = df["Allele"].astype(str).value_counts().reset_index()
        allele_counts.columns = ["Allele", "Count"]
        fig = px.bar(allele_counts, x="Allele", y="Count", text="Count", title="Allele counts")
        st.plotly_chart(add_bar_labels(fig), use_container_width=True)

    if "Allele" in df.columns and "Tier" in df.columns:
        pass_by_allele = (
            df[df["Tier"].astype(str) == "Pass"]
            .groupby("Allele", as_index=False)
            .size()
            .rename(columns={"size": "Pass Count"})
            .sort_values("Pass Count", ascending=False)
        )
        if not pass_by_allele.empty:
            fig = px.bar(
                pass_by_allele,
                x="Allele",
                y="Pass Count",
                text="Pass Count",
                title="Pass count by allele",
            )
            st.plotly_chart(add_bar_labels(fig), use_container_width=True)

    if "IC50 MT" in df.columns:
        top_ic50 = df.copy()
        ic50_col = numeric_column(top_ic50, "IC50 MT", "IC50 MT numeric")
        top_ic50 = top_ic50.dropna(subset=[ic50_col]).sort_values(ic50_col).head(25)
        if not top_ic50.empty:
            label_col = "Best Peptide" if "Best Peptide" in top_ic50.columns else "ID"
            fig = px.bar(
                top_ic50,
                x=label_col,
                y=ic50_col,
                text=ic50_col,
                title="Top 25 candidates by lowest IC50 MT",
            )
            st.plotly_chart(add_bar_labels(fig, "%{text:.2f}"), use_container_width=True)

    percentile_cols = [col for col in df.columns if "%ile MT" in str(col)]
    if percentile_cols:
        selected_percentile = st.selectbox("Selected percentile column", percentile_cols)
        percentile_frame = df.copy()
        selected_numeric = numeric_column(percentile_frame, selected_percentile, "Selected percentile numeric")

        threshold_rows = []
        for col in percentile_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            threshold_rows.extend(
                [
                    {"Percentile column": col, "Threshold": "<= 2", "Rows": int((values <= 2).sum())},
                    {"Percentile column": col, "Threshold": "<= 0.5", "Rows": int((values <= 0.5).sum())},
                ]
            )

        threshold_df = pd.DataFrame(threshold_rows)
        if not threshold_df.empty:
            fig = px.bar(
                threshold_df,
                x="Percentile column",
                y="Rows",
                color="Threshold",
                barmode="group",
                text="Rows",
                title="Percentile threshold summaries",
            )
            st.plotly_chart(add_bar_labels(fig), use_container_width=True)

        plot_df = percentile_frame.dropna(subset=[selected_numeric])
        if not plot_df.empty:
            fig = px.histogram(
                plot_df,
                x=selected_numeric,
                nbins=60,
                title=f"{selected_percentile} distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            if "Allele" in plot_df.columns:
                median_by_allele = (
                    plot_df.groupby("Allele", as_index=False)[selected_numeric]
                    .median()
                    .rename(columns={selected_numeric: f"Median {selected_percentile}"})
                    .sort_values(f"Median {selected_percentile}")
                )
                fig = px.bar(
                    median_by_allele,
                    x="Allele",
                    y=f"Median {selected_percentile}",
                    text=f"Median {selected_percentile}",
                    title=f"Median {selected_percentile} by allele",
                )
                st.plotly_chart(add_bar_labels(fig, "%{text:.3f}"), use_container_width=True)

            if "Tier" in plot_df.columns:
                fig = px.box(
                    plot_df,
                    x="Tier",
                    y=selected_numeric,
                    points="all",
                    title=f"{selected_percentile} by tier",
                )
                st.plotly_chart(fig, use_container_width=True)

            label_col = "Best Peptide" if "Best Peptide" in plot_df.columns else "ID"
            top_percentile = plot_df.sort_values(selected_numeric).head(50)
            fig = px.bar(
                top_percentile,
                x=label_col,
                y=selected_numeric,
                text=selected_numeric,
                title=f"Top 50 candidates by lowest {selected_percentile}",
            )
            st.plotly_chart(add_bar_labels(fig, "%{text:.3f}"), use_container_width=True)

        st.download_button(
            "Download percentile threshold summary TSV",
            threshold_df.to_csv(sep="\t", index=False),
            file_name="percentile_threshold_summary.tsv",
            mime="text/tab-separated-values",
        )

    summary_text = "\n".join(summary_lines)
    st.header("Summary")
    st.text_area("Summary", summary_text, height=350)
    download_text("Download summary text", summary_text, "pvacbind_summary.txt")


def show_rna_expression(df, rna_upload):
    st.header("RNA Expression")

    if rna_upload is None:
        st.info("Upload gene_abundance.tsv in the sidebar to use the RNA section.")
        st.stop()

    rna = read_uploaded_tsv(rna_upload, "gene_abundance.tsv")
    require_columns(df, ["ID"], "The pVACbind file")
    require_columns(rna, ["gene_name", "abundance"], "The RNA file")

    pvac = df.copy()
    pvac["Gene"] = pvac["ID"].apply(extract_gene_from_id)

    rna["abundance_numeric"] = pd.to_numeric(rna["abundance"], errors="coerce")
    rna_one_gene = (
        rna.groupby("gene_name", as_index=False)["abundance_numeric"]
        .max()
        .rename(columns={"gene_name": "Gene", "abundance_numeric": "RNAExpr"})
    )

    merged = pvac.merge(rna_one_gene, on="Gene", how="left")
    merged["RNAExpr_filled"] = merged["RNAExpr"].fillna(0)
    merged["log10_RNAExpr_plus1"] = np.log10(merged["RNAExpr_filled"] + 1)

    matched_rows = merged["RNAExpr"].notna().sum()
    missing_rows = merged["RNAExpr"].isna().sum()
    match_rate = matched_rows / len(merged) * 100 if len(merged) else 0

    unique_gene_expr = merged[["Gene", "RNAExpr"]].dropna().drop_duplicates()
    unique_source_genes_total = merged["Gene"].dropna().nunique()
    unique_source_genes_with_expr = unique_gene_expr["Gene"].nunique()

    median_expr = unique_gene_expr["RNAExpr"].median()
    mean_expr = unique_gene_expr["RNAExpr"].mean()
    genes_gt_0 = (unique_gene_expr["RNAExpr"] > 0).sum()
    genes_ge_1 = (unique_gene_expr["RNAExpr"] >= 1).sum()
    genes_ge_10 = (unique_gene_expr["RNAExpr"] >= 10).sum()
    genes_ge_34 = (unique_gene_expr["RNAExpr"] >= 34.038).sum()

    summary_lines = [
        f"Total pVACbind rows: {len(merged):,}",
        f"Rows with RNA expression matched: {matched_rows:,}",
        f"Rows missing RNA expression: {missing_rows:,}",
        f"RNA expression match rate: {match_rate:.2f}%",
        f"Unique source genes total: {unique_source_genes_total:,}",
        f"Unique source genes with RNA expression: {unique_source_genes_with_expr:,}",
        f"Median source gene RNA expression: {median_expr:.3f}",
        f"Mean source gene RNA expression: {mean_expr:.3f}",
        f"Genes with RNAExpr > 0: {genes_gt_0:,}",
        f"Genes with RNAExpr >= 1: {genes_ge_1:,}",
        f"Genes with RNAExpr >= 10: {genes_ge_10:,}",
        f"Genes with RNAExpr >= 34.038: {genes_ge_34:,}",
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows matched", f"{matched_rows:,}", f"{match_rate:.2f}%")
    col2.metric("Rows missing", f"{missing_rows:,}")
    col3.metric("Unique expressed genes", f"{unique_source_genes_with_expr:,}")
    col4.metric("Median RNAExpr", f"{median_expr:.3f}")

    st.header("RNA charts")

    expr_bins = pd.DataFrame(
        {
            "Category": ["RNAExpr > 0", "RNAExpr >= 1", "RNAExpr >= 10", "RNAExpr >= 34.038"],
            "Gene Count": [genes_gt_0, genes_ge_1, genes_ge_10, genes_ge_34],
        }
    )
    fig = px.bar(
        expr_bins,
        x="Category",
        y="Gene Count",
        text="Gene Count",
        title="Source genes above RNA expression thresholds",
    )
    st.plotly_chart(add_bar_labels(fig), use_container_width=True)

    if "Tier" in merged.columns:
        tier_expr = (
            merged.dropna(subset=["RNAExpr"])
            .groupby("Tier", as_index=False)["RNAExpr"]
            .median()
            .sort_values("RNAExpr", ascending=False)
        )
        tier_expr.columns = ["Tier", "Median RNAExpr"]
        if not tier_expr.empty:
            fig = px.bar(
                tier_expr,
                x="Tier",
                y="Median RNAExpr",
                text="Median RNAExpr",
                title="Median RNA expression by pVACbind tier",
            )
            st.plotly_chart(add_bar_labels(fig, "%{text:.2f}"), use_container_width=True)

            for _, row in tier_expr.iterrows():
                summary_lines.append(f"Median RNAExpr for {row['Tier']}: {row['Median RNAExpr']:.3f}")

    scatter_specs = [
        ("IC50 MT", "IC50 MT numeric", "RNA expression vs IC50 MT"),
        ("Pres %ile MT", "Pres %ile MT numeric", "RNA expression vs Pres %ile MT"),
    ]

    for source_col, numeric_col, title in scatter_specs:
        if source_col not in merged.columns:
            continue
        numeric_column(merged, source_col, numeric_col)
        corr_df = merged.dropna(subset=["RNAExpr", numeric_col])
        if len(corr_df) > 1:
            pearson_value = safe_corr(corr_df["RNAExpr"], corr_df[numeric_col], "pearson")
            spearman_value = safe_corr(corr_df["RNAExpr"], corr_df[numeric_col], "spearman")
            summary_lines.append(f"RNAExpr vs {source_col} Pearson correlation: {pearson_value:.4f}")
            summary_lines.append(f"RNAExpr vs {source_col} Spearman correlation: {spearman_value:.4f}")

            plot_df = corr_df.sample(min(5000, len(corr_df)), random_state=1)
            hover_cols = [col for col in ["Gene", "Best Peptide", "Allele", "Tier"] if col in plot_df.columns]
            fig = px.scatter(
                plot_df,
                x="RNAExpr",
                y=numeric_col,
                hover_data=hover_cols,
                title=title,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.header("Top expressed source genes")

    top_genes = unique_gene_expr.sort_values("RNAExpr", ascending=False).head(25)
    if not top_genes.empty:
        fig = px.bar(
            top_genes,
            x="Gene",
            y="RNAExpr",
            text="RNAExpr",
            title="Top 25 source genes by RNA expression",
        )
        st.plotly_chart(add_bar_labels(fig, "%{text:.2f}"), use_container_width=True)
    st.dataframe(top_genes, use_container_width=True)

    st.header("Merged table")
    st.dataframe(merged, use_container_width=True, height=600)

    st.download_button(
        "Download merged pVACbind + RNA TSV",
        merged.to_csv(sep="\t", index=False),
        file_name="pvacbind_with_rna_expression.tsv",
        mime="text/tab-separated-values",
    )

    summary_text = "\n".join(summary_lines)
    st.header("RNA Summary")
    st.text_area("Summary", summary_text, height=450)
    download_text("Download RNA summary text", summary_text, "rna_summary.txt")


st.title("pVACbind Viewer")

page = st.sidebar.radio(
    "Choose section",
    ["pVACbind Summary", "RNA Expression"],
)

st.sidebar.header("Upload files")
pvac_upload = st.sidebar.file_uploader("Upload pVACbind TSV", type=["tsv"])
rna_upload = st.sidebar.file_uploader("Optional: upload gene_abundance.tsv", type=["tsv"])

if pvac_upload is None:
    st.info("Upload an aggregated pVACbind TSV file to begin.")
    st.stop()

df = read_uploaded_tsv(pvac_upload, "pVACbind TSV")

if df.empty:
    st.warning("The uploaded pVACbind TSV has no rows.")
    st.stop()

filtered_df = build_sidebar_filters(df)

if page == "pVACbind Summary":
    show_pvacbind_summary(df, filtered_df)
elif page == "RNA Expression":
    show_rna_expression(df, rna_upload)
