def chart_lake_frequencies_old(info, title, location=lake_location):
    df = pd.melt(pd.DataFrame(info), "Iteration")
    df = df[df["Iteration"] % 10000 == 0 & ("S_Freq")]

    frequencies = []
    epsilons = []
    for i in range(df["Iteration"].min(), df["Iteration"].max(), 10000):
        f = np.array(df[(df["variable"] == "S_Freq") & (df.Iteration == i)]["value"])
        frequencies.append(f.sum(axis=0))
        e = np.array(df[(df["variable"] == "Epsilon") & (df.Iteration == i)]["value"])
        epsilons.append(e.mean())
    frequencies = np.array(frequencies)
    epsilons = np.array(epsilons)

    wcs = []
    sfs = []
    for i in range(1, len(frequencies)):
        freq = frequencies[i] - frequencies[i - 1]
        freq_sum = freq.sum()
        wait_cut = freq.sum(axis=0) / freq_sum
        wcs.append(wait_cut)
        sf = freq.sum(axis=1) / freq_sum
        sfs.append(sf)
        # print(f"{i:2}: {epsilons[i]:0.3f} || {wait_cut[0]:0.3f} {wait_cut[1]:0.3f} || {sf[0]:0.4f} {sf[1]:0.4f} {sf[2]:0.4f} {sf[3]:0.4f} {sf[4]:0.4f} {sf[5]:0.4f} {sf[6]:0.4f} ")

    sfs = np.array(sfs) * 100
    sfs = np.reshape(sfs, (4, 4))

    ax = sns.heatmap(
        sfs[-1],
        cmap="flare",
        cbar_kws={"format": "%.0f%%", "label": "Frequency Percent"},
    )
    # ax.set_ylabel("Iteration (10000)")
    ax.set_xlabel("State")
    ax.set_title(title)
    suptitle = "Lake Frequencies States"
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)
