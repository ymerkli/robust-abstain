"""
Modified from: https://github.com/locuslab/smoothing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math
import logging

from robustabstain.analysis.plotting.utils.helpers import find_smoothing_log
from robustabstain.utils.helpers import convert_floatstr


class Accuracy(object):
    def at_radii(self, radii: np.ndarray) -> None:
        raise NotImplementedError()


class CertAcc(Accuracy):
    def __init__(self, data_file_path: str, adv_norm: str = 'L2') -> None:
        self.data_file_path = data_file_path
        self.adv_norm = adv_norm

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter=",")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float) -> float:
        radius_col = 'l2_radius' if self.adv_norm == 'L2' else 'linf_radius'
        correct = df['label'] == df['prediction']
        certified = (df[radius_col] >= radius)
        return 100.0 * (certified & correct).mean()


class CertInacc(Accuracy):
    def __init__(self, data_file_path: str, adv_norm: str = 'L2') -> None:
        self.data_file_path = data_file_path
        self.adv_norm = adv_norm

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter=",")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float) -> float:
        radius_col = 'l2_radius' if self.adv_norm == 'L2' else 'linf_radius'
        inaccurate = (df['label'] != df['prediction']) & (df['prediction'] != -1)
        certified = (df[radius_col] >= radius)
        return 100.0 * (certified & inaccurate).mean()


class CommitPrec(Accuracy):
    def __init__(self, data_file_path: str, adv_norm: str = 'L2') -> None:
        self.data_file_path = data_file_path
        self.adv_norm = adv_norm

    def at_radii(self, radii: np.ndarray) -> None:
        df = pd.read_csv(self.data_file_path, delimiter=",")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float) -> float:
        radius_col = 'l2_radius' if self.adv_norm == 'L2' else 'linf_radius'
        correct = df['label'] == df['prediction']
        certified = (df[radius_col] >= radius)
        if (certified).mean() == 0:
            commit_prec = 0
        else:
            commit_prec = 100.0 * (certified & correct).mean() / (certified).mean()
        return commit_prec


class CommitRate(Accuracy):
    def __init__(self, data_file_path: str, adv_norm: str = 'L2') -> None:
        self.data_file_path = data_file_path
        self.adv_norm = adv_norm

    def at_radii(self, radii: np.ndarray) -> None:
        df = pd.read_csv(self.data_file_path, delimiter=",")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float) -> float:
        radius_col = 'l2_radius' if self.adv_norm == 'L2' else 'linf_radius'
        certified = (df[radius_col] >= radius)
        commit_rate = 100.0 * (certified).mean()
        return commit_rate

class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_cert_acc(
        outfile: str, title: str, max_radius: float,
        lines: List[Line], radius_step: float = 0.01
    ) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 100))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("Radius", fontsize=16)
    plt.ylabel("Certified Accuracy [%]", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=11)
    plt.grid(True)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()
    logging.info(f'Exported plot to {outfile}[.png/.pdf]')


def plot_cert_inacc(
        outfile: str, title: str, max_radius: float,
        lines: List[Line], radius_step: float = 0.01
    ) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 30))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("Radius", fontsize=16)
    plt.ylabel("Certified Inaccurate [%]", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=11)
    plt.grid(True)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()
    logging.info(f'Exported plot to {outfile}[.png/.pdf]')


def plot_commit_prec(
        outfile: str, title: str, max_radius: float,
        lines: List[Line], radius_step: float = 0.01
    ) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((80, 100))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("Radius", fontsize=16)
    plt.ylabel("Commit Precision [%]", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=11)
    plt.grid(True)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()
    logging.info(f'Exported plot to {outfile}[.png/.pdf]')


def plot_commit_rate(
        outfile: str, title: str, max_radius: float,
        lines: List[Line], radius_step: float = 0.01
    ) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((80, 100))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("Radius", fontsize=16)
    plt.ylabel("Commit Rate [%]", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=11)
    plt.grid(True)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()
    logging.info(f'Exported plot to {outfile}[.png/.pdf]')



def plot_commit_prec_rate(
        outfile: str, title: str, max_radius: float,
        lines: List[Line], radius_step: float = 0.01
    ) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((80, 100))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("Radius", fontsize=16)
    plt.ylabel("Commit Precision [%]", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=11)
    plt.grid(True)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()
    logging.info(f'Exported plot to {outfile}[.png/.pdf]')
