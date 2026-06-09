import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="pVACbind Viewer", layout="wide")

st.title("pVACbind Viewer")

page = st.sidebar.radio(
    "Choose section",
    ["pVACbind Summary", "RNA Expression"]
)

st.sidebar.header("Upload files")

pvac_upload = st.sidebar.file_uploader(
    "Upload aggregated pVACbind TSV",
    type=["tsv"]
)

rna_upload = st.sidebar.file_uploader(
    "Optional: upload gene_abundance.tsv",
    type=["tsv"]
)

if pvac_upload is None:
    st.info("Upload an aggregated pVACbind TSV file to begin.")
    st.stop()

df = pd.read_csv(pvac_upload, sep="\t", low_memory=False)

def extract_gene_from_id(x):
    if pd.isna(x):
        return np.nan
    return str(x).split("|")[0]

if page == "pVACbind Summary":
    st.header("pVACbind Summary")

    st.write(f"Rows: {df.shape[0]:,}")
    st.write(f"Columns: {df.shape[1]:,}")

    filtered = df.copy()

    st.sidebar.header("Filters")

    if "Tier" in filtered.columns:
        tiers = sorted(filtered["Tier"].dropna().astype(str).unique())
        selected_tiers = st.sidebar.multiselect("Tier", tiers, default=tiers)
        filtered = filtered[filtered["Tier"].astype(str).isin(selected_tiers)]

    if "Allele" in filtered.columns:
        alleles = sorted(filtered["Allele"].dropna().astype(str).unique())
        selected_alleles = st.sidebar.multiselect("Allele", alleles, default=alleles)
        filtered = filtered[filtered["Allele"].astype(str).isin(selected_alleles)]

    if "IC50 MT" in filtered.columns:
        filtered["IC50 MT numeric"] = pd.to_numeric(filtered["IC50 MT"], errors="coerce")
        max_ic50 = st.sidebar.number_input("Max IC50 MT", value=50000.0)
        filtered = filtered[filtered["IC50 MT numeric"] <= max_ic50]

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

    st.subheader("Filtered table")
    st.write(f"Filtered rows: {filtered.shape[0]:,}")
    st.dataframe(filtered, use_container_width=True, height=600)

    st.download_button(
        "Download filtered TSV",
        filtered.to_csv(sep="\t", index=False),
        file_name="filtered_pvacbind.tsv",
        mime="text/tab-separated-values"
    )

    summary_lines = []

    total_rows = len(df)
    summary_lines.append(f"Total rows: {total_rows:,}")
    summary_lines.append(f"Total columns: {df.shape[1]:,}")
    summary_lines.append(f"Filtered rows shown: {len(filtered):,}")

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
        unique_peptides = df["Best Peptide"].nunique()
        summary_lines.append(f"Unique Best Peptide values: {unique_peptides:,}")

    if "Allele" in df.columns:
        unique_alleles = df["Allele"].nunique()
        summary_lines.append(f"Unique alleles: {unique_alleles:,}")

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

    st.header("Charts")

    if "Tier" in df.columns:
        tier_counts = df["Tier"].astype(str).value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]

        fig = px.bar(
            tier_counts,
            x="Tier",
            y="Count",
            text="Count",
            title="Tier counts"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    if "Allele" in df.columns:
        allele_counts = df["Allele"].astype(str).value_counts().reset_index()
        allele_counts.columns = ["Allele", "Count"]

        fig = px.bar(
            allele_counts,
            x="Allele",
            y="Count",
            text="Count",
            title="Allele counts"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    if "IC50 MT" in df.columns:
        top_ic50 = df.copy()
        top_ic50["IC50 MT numeric"] = pd.to_numeric(top_ic50["IC50 MT"], errors="coerce")
        top_ic50 = top_ic50.dropna(subset=["IC50 MT numeric"]).sort_values("IC50 MT numeric").head(25)

        label_col = "Best Peptide" if "Best Peptide" in top_ic50.columns else "ID"

        fig = px.bar(
            top_ic50,
            x=label_col,
            y="IC50 MT numeric",
            text="IC50 MT numeric",
            title="Top 25 candidates by lowest IC50 MT"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Summary")
    st.text_area("Summary", "\n".join(summary_lines), height=350)

if page == "RNA Expression":
    st.header("RNA Expression")

    if rna_upload is None:
        st.info("Upload gene_abundance.tsv in the sidebar to use the RNA section.")
        st.stop()

    rna = pd.read_csv(rna_upload, sep="\t", low_memory=False)

    if "ID" not in df.columns:
        st.error("The pVACbind file needs an ID column so the app can extract source genes.")
        st.stop()

    if "gene_name" not in rna.columns or "abundance" not in rna.columns:
        st.error("The RNA file needs columns named gene_name and abundance.")
        st.stop()

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

    median_expr = unique_gene_expr["RNAExpr"].median()
    mean_expr = unique_gene_expr["RNAExpr"].mean()

    genes_gt_0 = (unique_gene_expr["RNAExpr"] > 0).sum()
    genes_ge_1 = (unique_gene_expr["RNAExpr"] >= 1).sum()
    genes_ge_10 = (unique_gene_expr["RNAExpr"] >= 10).sum()
    genes_ge_34 = (unique_gene_expr["RNAExpr"] >= 34.038).sum()

    summary_lines = []
    summary_lines.append(f"Total pVACbind rows: {len(merged):,}")
    summary_lines.append(f"Rows with RNA expression matched: {matched_rows:,}")
    summary_lines.append(f"Rows missing RNA expression: {missing_rows:,}")
    summary_lines.append(f"RNA expression match rate: {match_rate:.2f}%")
    summary_lines.append(f"Unique source genes with RNA expression: {unique_gene_expr['Gene'].nunique():,}")
    summary_lines.append(f"Median source gene RNA expression: {median_expr:.3f}")
    summary_lines.append(f"Mean source gene RNA expression: {mean_expr:.3f}")
    summary_lines.append(f"Genes with RNAExpr > 0: {genes_gt_0:,}")
    summary_lines.append(f"Genes with RNAExpr >= 1: {genes_ge_1:,}")
    summary_lines.append(f"Genes with RNAExpr >= 10: {genes_ge_10:,}")
    summary_lines.append(f"Genes with RNAExpr >= 34.038: {genes_ge_34:,}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows matched", f"{matched_rows:,}", f"{match_rate:.2f}%")
    col2.metric("Rows missing", f"{missing_rows:,}")
    col3.metric("Unique expressed genes", f"{unique_gene_expr['Gene'].nunique():,}")
    col4.metric("Median RNAExpr", f"{median_expr:.3f}")

    st.header("RNA charts")

    expr_bins = pd.DataFrame({
        "Category": ["RNAExpr > 0", "RNAExpr >= 1", "RNAExpr >= 10", "RNAExpr >= 34.038"],
        "Gene Count": [genes_gt_0, genes_ge_1, genes_ge_10, genes_ge_34]
    })

    fig = px.bar(
        expr_bins,
        x="Category",
        y="Gene Count",
        text="Gene Count",
        title="Source genes above RNA expression thresholds"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    if "Tier" in merged.columns:
        tier_expr = (
            merged.dropna(subset=["RNAExpr"])
            .groupby("Tier", as_index=False)["RNAExpr"]
            .median()
            .sort_values("RNAExpr", ascending=False)
        )
        tier_expr.columns = ["Tier", "Median RNAExpr"]

        fig = px.bar(
            tier_expr,
            x="Tier",
            y="Median RNAExpr",
            text="Median RNAExpr",
            title="Median RNA expression by pVACbind tier"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        for _, row in tier_expr.iterrows():
            summary_lines.append(f"Median RNAExpr for {row['Tier']}: {row['Median RNAExpr']:.3f}")

    if "IC50 MT" in merged.columns:
        merged["IC50 MT numeric"] = pd.to_numeric(merged["IC50 MT"], errors="coerce")
        corr_df = merged.dropna(subset=["RNAExpr", "IC50 MT numeric"])

        if len(corr_df) > 1:
            pearson_ic50 = corr_df["RNAExpr"].corr(corr_df["IC50 MT numeric"], method="pearson")
            spearman_ic50 = corr_df["RNAExpr"].corr(corr_df["IC50 MT numeric"], method="spearman")

            summary_lines.append(f"RNAExpr vs IC50 MT Pearson correlation: {pearson_ic50:.4f}")
            summary_lines.append(f"RNAExpr vs IC50 MT Spearman correlation: {spearman_ic50:.4f}")

            plot_df = corr_df.sample(min(5000, len(corr_df)), random_state=1)

            fig = px.scatter(
                plot_df,
                x="RNAExpr",
                y="IC50 MT numeric",
                hover_data=["Gene"],
                title="RNA expression vs IC50 MT"
            )
            st.plotly_chart(fig, use_container_width=True)

    if "Pres %ile MT" in merged.columns:
        merged["Pres %ile MT numeric"] = pd.to_numeric(merged["Pres %ile MT"], errors="coerce")
        corr_df = merged.dropna(subset=["RNAExpr", "Pres %ile MT numeric"])

        if len(corr_df) > 1:
            pearson_pres = corr_df["RNAExpr"].corr(corr_df["Pres %ile MT numeric"], method="pearson")
            spearman_pres = corr_df["RNAExpr"].corr(corr_df["Pres %ile MT numeric"], method="spearman")

            summary_lines.append(f"RNAExpr vs Pres %ile MT Pearson correlation: {pearson_pres:.4f}")
            summary_lines.append(f"RNAExpr vs Pres %ile MT Spearman correlation: {spearman_pres:.4f}")

            plot_df = corr_df.sample(min(5000, len(corr_df)), random_state=1)

            fig = px.scatter(
                plot_df,
                x="RNAExpr",
                y="Pres %ile MT numeric",
                hover_data=["Gene"],
                title="RNA expression vs Pres %ile MT"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.header("Top expressed source genes")

    top_genes = unique_gene_expr.sort_values("RNAExpr", ascending=False).head(25)

    fig = px.bar(
        top_genes,
        x="Gene",
        y="RNAExpr",
        text="RNAExpr",
        title="Top 25 source genes by RNA expression"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(top_genes, use_container_width=True)

    st.header("Merged table")
    st.dataframe(merged, use_container_width=True, height=600)

    st.download_button(
        "Download merged pVACbind + RNA TSV",
        merged.to_csv(sep="\t", index=False),
        file_name="pvacbind_with_rna_expression.tsv",
        mime="text/tab-separated-values"
    )

    st.header("RNA Summary")
    st.text_area("RNA Summary", "\n".join(summary_lines), height=450)