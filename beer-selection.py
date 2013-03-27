#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: Seth Brown
Description: Beer Selection Analysis
Date: 2013-03-24

Python 2.7x
"""
from __future__ import division
import sys
from itertools import groupby, chain
from collections import Counter, namedtuple
import pandas as pd
import numpy as np
from numpy.random.mtrand import dirichlet
import scipy as sp
import matplotlib.pyplot as plt


def maybe_float(s):
    """ Attempt to convert a string to a float
    """
    try:
        val = float(s)
    except ValueError:
        val = s

    return val


def normalize_scale(s):
    """ Normalize a rating string, s to a 5 points scale, (0-5]

    Parameters:
    --------------
    s: rating string, eg. '8/20', '10', etc.

    Output:
    -----------
    normalized score: float, (0-5]
    """
    if '/' in s:
        num, denom = map(int, s.split('/'))
        fctr = float(denom) / 5
        score = num / fctr
    else:
        score = s

    return 0.5 * round(float(score) / 0.5)


def abv_group(abv):
    """ Convert ABV into a common categories
    """
    if 0 <= abv < 5:
        abv_cat = '0-5%'
    elif 5 <= abv < 10:
        abv_cat = '5-10%'
    elif 10 <= abv < 15:
        abv_cat = '10-15%'
    elif 15 <= abv < 20:
        abv_cat = '15-20%'
    else:
        abv_cat = '>20%'

    return abv_cat


def ratings_pmf(df, col_key):
    """ Construct a PMF from a Pandas DataFrame column
    """
    pmf = lambda count, total: count / float(total)
    values = df.groupby(col_key).size()
    prs = values.apply(pmf, total=values.sum())

    return prs


def beer_df(lines):
    """ Process beer data into a Pandas dataframe

        Parameters
        ------------------
        lines: generator of lines from beer data file

        Output
        --------------
        dataframe: cleaned and normalized data. cols:
             Beer_id, Brewer_id, Abv, Style, Appear., Aroma,
             Palate, Taste, Overall, Time, Profile_name
    """
    cols = ('Beer_id', 'Brewer_id', 'Abv', 'Style', 'Appear.', 'Aroma',
            'Palate', 'Taste', 'Overall', 'Time', 'Profile_name')
    data = []
    txt = 'review/text'
    for (key, group) in groupby(lines, lambda t: t.startswith('beer/name')):
        if key:
            next(group).strip()
        else:
            rating_data = [i.strip() for i in group if not i.startswith(txt)]
            abv = rating_data[2]
            # some abv fields are malformed--ensure abv contains decimal pt.
            if '.' in abv:
                # remove blank lines
                rating_data = filter(lambda l: l != '', rating_data)
                rating_data = [i.split(':') for i in rating_data]
                rating_data = [i[1].strip() for i in rating_data]
                scale = map(normalize_scale, rating_data[4:9])
                rating_data = rating_data[:4] + scale + rating_data[9:]
                rating_data = [maybe_float(s) for s in rating_data]
                if len(rating_data) == 11:
                    abv = abv_group(rating_data[2])
                    data.append(rating_data)

    return pd.DataFrame(data, columns=cols)


def correlation_matrix(dataframe, filename, bar_label=''):
    """ Plot a correlation matrix from a Pandas DataFrame
    """
    plt.figure(figsize=(7, 9))
    col_labels = dataframe.columns.tolist()
    tick_indices = np.arange(0.5, len(col_labels) + 0.5)
    plt.pcolormesh(dataframe.values, cmap='RdBu', vmin=-1, vmax=1)
    cbar = plt.colorbar(drawedges=False, orientation='horizontal')
    cbar.solids.set_edgecolor('face')
    cbar.set_label('Spearman\'s ' + r'$\rho$')
    plt.xticks(tick_indices, col_labels)
    plt.yticks(tick_indices, col_labels)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def pmfs(df, filename):
    """ Plot overall and taste distributions
    """
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    idx = sp.arange(0, 5.5, 0.5)[::-1]
    taste_prs = ratings_pmf(df, 'Taste')
    ax1.bar(idx, [0] + list(taste_prs.values), width=0.5, align='center')
    ax1.set_xlim([-0.5, 5.5])
    ax1.set_xticklabels([6, 5, 4, 3, 2, 1, 0])
    ax1.set_xlabel('Taste Ratings')
    ax1.set_ylabel('Probability\n')
    overall_prs = ratings_pmf(df, 'Overall')
    ax2.bar(idx, overall_prs.values, width=0.5, align='center')
    ax2.set_xlabel('Overall Ratings')
    ax2.set_ylabel('Probability\n')
    plt.subplots_adjust(right=1.2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def filter_groups(df):
    data = {}
    for k, v in df.iterrows():
        key = (v['Abv'], v['Style'])
        data.setdefault(key, []).append(v['Overall'])
    # exclude groups with < 30 representative beers & < 2000 reviews
    data = {k: [len(v)] + list(chain(v)) for k, v in data.items()}
    data = {k: v[1:] for k, v in data.items() if v[0] > 30}
    data = {k: v for k, v in data.items() if len(v) > 2000}
    data = {k: map(round, v) for k, v in data.items()}

    return data


def prob5(data, iters=5000):
    post = []
    post_temp = namedtuple('post', 'style, abv, post_prob')
    for label, vals in data.items():
        counts = Counter(vals)
        freq_data = [(i, j) for i, j in counts.items()]
        keys = [i[0] for i in freq_data]
        obs = [i[1] for i in freq_data]
        results = []
        for n in xrange(iters):
            samp = dirichlet([x for x in obs])
            results.append(samp)
        results = np.array(results)
        probs = np.mean(results, axis=0)
        n_data = zip(keys, probs)
        for key, prob in n_data:
            if prob is not None and key == 5:
                datum = post_temp(style=label[1], abv=label[0], post_prob=prob)
                post.append(datum)

    return post


def terminal_plot(data, filename, n=10):
    """ Plot the terminal (highest/lowest) posterior probability ratings
    """
    label_strs = [' '.join((i.style, i.abv)) for i in data]
    labels = [unicode(i.decode('latin-1')) for i in label_strs]
    probs = [i.post_prob for i in data]
    idx = range(n)
    fig, ((ax1, ax2)) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    ax1.yaxis.get_majorticklocs()
    ax1.yaxis.set_ticks(idx)
    ax1.set_yticklabels(labels[:n], fontsize=15)
    ax2.yaxis.get_majorticklocs()
    ax2.yaxis.set_ticks(idx)
    ax2.set_yticklabels(labels[-n:], fontsize=15)
    ax2.set_xlabel('Probability')
    ax1.barh(idx, probs[:n], height=0.8, align='center')
    ax2.barh(idx, probs[-n:], height=0.8, align='center')
    plt.subplots_adjust(hspace=100)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def main():
    lines = (line for line in sys.stdin)
    df = beer_df(lines)
    corr = df.corr(method='spearman')
    corr_df = corr.ix[2:-1, 2:-1]
    correlation_matrix(corr_df, 'beer-attributes.svg')
    pmfs(df, 'overall-ratings-pmf.svg')
    df['Abv'] = df['Abv'].map(lambda t: abv_group(t))
    data = filter_groups(df)
    post = prob5(data)
    groups = sorted([i for i in post], key=lambda t: t.post_prob)
    terminal_plot(groups, 'abv-probs.svg')

if __name__ == '__main__':
    main()
