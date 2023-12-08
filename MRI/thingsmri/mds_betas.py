"""
Run multidimensional scaling on single trial responses in category-selective brain areas. 

Usage:
python mds_betas.py <subject_ID> <bids_path> <category_1> <category_2> <color_1> <color_2>

Examples:
python mds_betas.py 01 /home/user/thingsmri vehicle tool lightskyblue mediumvioletred
python mds_betas.py 01 /home/user/thingsmri animal food rebeccapurple mediumspringgreen
"""

from os.path import join as pjoin
import numpy as np
import os
from nilearn.masking import intersect_masks
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import MDS

sys.path.append(os.getcwd())
from thingsmri.utils import load_category_df, get_category_rois
from thingsmri.betas import (
    load_betas,
    load_filenames,
    filter_catch_trials,
    average_betas_per_concept,
)


def run_mds_on_betas(
    sub,
    bidsroot,
    mask,
    betas_derivname="betas_loo/on_residuals/scalematched",
    out_derivname="mds",
    target_cats=["body part", "plant"],
    target_colors=["rebeccapurple", "mediumspringgreen"],
    mds_kws=dict(
        n_components=2,
        n_init=10,
        max_iter=5_000,
        n_jobs=-1,
        dissimilarity="precomputed",
    ),
    seed=0,
):
    """
    Example:
        sub = sys.argv[1]
        bidsroot = pjoin(os.pardir, os.pardir, os.pardir)
        # get category selective areas as a mask
        julian_basedir = pjoin(
            bidsroot, 'derivatives', 'julian_parcels', 'julian_parcels_edited'
        )
        julian_dir = pjoin(julian_basedir, f'sub-{sub}')
        roi_files = glob.glob(pjoin(julian_dir, '*', '*.nii.gz'))
        roi_files = [rf for rf in roi_files if 'RSC' not in rf]  # dismiss RSC
        loc_files = glob.glob(pjoin(julian_dir.replace('edited', 'intersected'),
                                    'object_parcels', '*.nii.gz'))
        roi_files += loc_files
        mask = intersect_masks(roi_files, threshold=0, connected=False)
        for target_cats, target_colors in tqdm(zip(
            [['animal', 'food'], ['body part', 'plant']],
            [['lightskyblue', 'mediumvioletred'], ['rebeccapurple', 'mediumspringgreen']],
        ), desc='target_cats', total=2):
            run_mds_on_betas(sub, bidsroot, mask=mask, target_cats=target_cats, target_colors=target_colors)
    """
    np.random.seed(seed)
    # if colors are passed as RBG tuples, normalize to 0-1
    for i, color in enumerate(target_colors):
        if type(color) == tuple:
            target_colors[i] = tuple([e / 255 for e in color])
    # define file names
    outdir = pjoin(bidsroot, "derivatives", out_derivname, f"sub-{sub}")
    out_npy = pjoin(
        outdir,
        f'{target_cats[0]}_{target_cats[1]}_ninit-{mds_kws["n_init"]}_maxiter-{mds_kws["max_iter"]}.npy',
    )
    out_png = out_npy.replace(".npy", ".png")
    out_pdf = out_npy.replace(".npy", ".pdf")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # load betas
    betas_ = load_betas(sub, mask, bidsroot)
    fnames_ = load_filenames(sub, bidsroot, betas_derivname)
    # exclude catch trials
    betas_, fnames_, noncatch_is = filter_catch_trials(betas_, fnames_)
    # average within concepts
    betas, concepts = average_betas_per_concept(betas_, fnames_)
    # compute correlation distance
    rdm = 1 - np.corrcoef(betas)
    # load category names
    cat_df = load_category_df()
    cats = []
    for con in concepts:
        if con[-1] in [
            "1",
            "2",
        ]:  # some concepts are coded 'bracelet2' or 'bow1', we just count them as individuals
            con = con[:-1]
        row = cat_df.loc[cat_df["Word"] == con.replace("_", " ")]
        hits = [cat for cat in target_cats if row[cat].values[0] == 1]
        result = (
            hits[0] if len(hits) else "Other"
        )  # only keep the first category found, or "Other"
        cats.append(result)
    cats = np.array(cats)
    sort = np.argsort(cats)
    cats = cats[sort]

    # show how many exemplars were found per category
    # uniquecats, counts = np.unique(cats, return_counts=True)
    # run MDS and save embedding
    mds = MDS(random_state=seed, **mds_kws)
    mds.fit(rdm)
    Y = mds.embedding_[sort]
    np.save(out_npy, Y)
    # plot and save
    colors_ = target_colors + ["lightgrey"]
    labels_ = target_cats + ["Other"]
    fig = plt.figure(figsize=(7, 7))
    g = sns.scatterplot(
        x=Y[:, 0],
        y=Y[:, 1],
        hue=cats,
        hue_order=labels_,
        palette=colors_,
        s=300,
        alpha=0.7,
        linewidth=0,
        legend=False,
    )
    plt.axis("off")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    return None


if __name__ == "__main__":
    sub, bidsroot, cat1, cat2, col1, col2 = sys.argv[1:]
    rois = get_category_rois(sub, bidsroot, "/rois/category_localizer")
    for roiname, roifile in rois:
        if "RSC" in roiname:
            del rois[roiname]
    mask = intersect_masks(rois.values(), threshold=0, connected=False)
    run_mds_on_betas(
        sub, bidsroot, mask=mask, target_cats=[cat1, cat2], target_colors=[col1, col2]
    )
