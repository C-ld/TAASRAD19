import gzip
import os
import tarfile

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from settings import (
    CROP_SIZE,
    GAP_LENGTH,
    IMG_SIZE,
    MIN_RUN_SIZE,
    SKIP_ROWS,
    PARSE_OFFSET,
    MASK,
    PARSED_TYPE,
    RUN_SIZE,
    VMIN,
    VMAX
)


class BadAsciiException(Exception):
    """Raised when the shape size is unexpected or the ascii file is malformed"""


def parse_ascii(
    fileobj,
    offset=PARSE_OFFSET,
    scan_size=IMG_SIZE,
    skip_header=SKIP_ROWS,
    output_type=PARSED_TYPE,
):
    try:
        arr = np.genfromtxt(
            fileobj,
            skip_footer=offset,
            skip_header=skip_header + offset,
            delimiter="\t",
            autostrip=True,
            usecols=range(offset, scan_size - offset),
            dtype=np.float32,
        )
        arr[np.where(arr == (-99.0))] = 0.0
        arr[np.where(arr == (55.0))] = 0.0
        if arr.shape != (scan_size - offset * 2, scan_size - offset * 2):
            raise BadAsciiException("{} has wrong shape".format(fileobj.name))
        
        if output_type:
            arr = arr.astype(output_type)

        return arr
    except:
        raise BadAsciiException("{} is broken".format(fileobj.name))
    finally:
        fileobj.close()


def recursively_check_run_consistency(f, run, last):
    # runs = []
    if len(run) == RUN_SIZE:
        try:
            well_formed_scans = []
            for run_scan in run:
                with f.extractfile(run_scan) as extr, gzip.GzipFile(
                    fileobj=extr
                ) as gfile:
                    data = parse_ascii(gfile) #data: 480 * 480
                    
                    ### test

                    # fig = plt.figure(figsize=(4.8, 4.8))
                    # ax = plt.axes()
                    # ax.set_axis_off()
                    # alpha = data / 1
                    # alpha[alpha < 1] = 0
                    # alpha[alpha > 1] = 1
                    # ax.imshow(data, alpha=alpha, cmap="viridis")
                    # ## fig = plt.hist(data, range = (data.min(), data.max()))

                    # plt.savefig('./{}.png'.format("test"))
                    # plt.close()

                    ### endtest

                    well_formed_scans.append(data)
        # Note: the parse_ascii function raises a BadAsciiException when the radar scan is malformed.
        # When this happens: the scans to the left are a run and are well-formed. They are saved if they are
        # long enough. We know the scans to the right are a run, but not if they are well-formed.
        except BadAsciiException:
            print("BadAsciiException at time:", last)
        # else:
            # runs.append(
            #     (
            #         np.stack(well_formed_scans),
            #         (last - GAP_LENGTH * (len(run) - 1), last),
            #     )
            # )
    return well_formed_scans

def save_plots(field, labels, res_path, figsize=None,
                       vmin=0, vmax=10, cmap="viridis", npy=False, **imshow_args):

    for i, data in enumerate(field):
        
        # fig = plt.figure(figsize=figsize)
        # ax = plt.axes()
        # ax.set_axis_off()
        # alpha = data / 1
        # alpha[alpha < 1] = 0
        # alpha[alpha > 1] = 1

        # img = ax.imshow(data, alpha=alpha, vmin=vmin, vmax=vmax, **imshow_args)
        # plt.savefig('{}/{}.png'.format(res_path, labels[i]))
        # plt.close()
        im = Image.fromarray(data.astype(np.uint8))
        im.save('{}/{}.png'.format(res_path, labels[i]))
        # im.getdata()
        im.close()
        if npy:
            with open( '{}/{}.npy'.format(res_path, labels[i]), 'wb') as f:
                np.save(f, data)

def identify_runs(f, scans, tags, dpath):
    #   LD: read data in one day and collect

    # select only the MAX Z product
    scans = [scan for scan in scans if "cmaZ" in scan.name]
    if scans:
        # A run is sequence of consecutive radar scans within GAP_LENGTH time.
        # This list will contain tuples whose first element is the run data
        # and the second element is a tuple of  (start_datetime, end_datetime)
        # for that run.
        runs = []
        cnt = 0

        # Run length counts the number of consecutive scans within GAP_LENGTH time.
        run_length = 0

        # First date in scans
        last = pd.to_datetime(scans[0].name[13:-9])

        def check_run(run):
            mean = np.mean(run)
            if mean >= 1:
                return True
            elif mean >= 0.5 and tags != "":
                return True
            else:
                return False

        for idx, scan in enumerate(scans):
            if ".ascii.gz" in scan.name:
                scan_time = pd.to_datetime(scan.name[13:-9])
                #print(scan_time)

                # If scan is equal to GAP_LENGTH of last scan, increase run_length window
                if scan_time - last == GAP_LENGTH:
                    run_length += 1
                elif scan_time - last > GAP_LENGTH:
                    run_length = 1
                last = scan_time

                #if lenth == RUN_SIZE (29), save and make a new run 
                if run_length == RUN_SIZE:
                    run = recursively_check_run_consistency(
                        f, scans[idx - run_length + 1 : idx + 1], last
                    )
                    # run.shape = (29, 480, 480)
                    run_length = 0
                    if check_run(run):
                        #T1 create dir
                        path = os.path.join(dpath, '{:0>2d}'.format(cnt))
                        os.makedirs(path, exist_ok=True)
                        #T2 for each png, save and close
                        save_plots(run, figsize=(IMG_SIZE/100.0, IMG_SIZE/100.0),
                            labels=['{:0>2d}'.format(i) for i in range(RUN_SIZE)],
                            res_path=path, vmin=VMIN, vmax=VMAX)
                        cnt += 1

        # if run_length == MIN_RUN_SIZE:
        #     runs += recursively_check_run_consistency(f, scans[-run_length:], last)

        ## set to 0 all the points outside the radar circle

        # for run, _ in runs:
        #     run[:, ~MASK] = 0

        # iterate through all the runs and remove those without enough rainfall
        # discard runs with mean < 0.5 dbz
        # discard runs with mean < 1 dbz and without any weak label


        # runs = [(run, periods) for run, periods in runs if check_run(run)]

        # return runs if runs else None


def worker(args):
    day, radar_directory, ouput_dir, tags = args
    # add output /yyyy/yyyydd directory
    dpath = os.path.join(ouput_dir, str(day.year), day.strftime("%Y%m%d"))
    # if not os.path.exists(dpath): 
    #     os.makedirs(dpath, exist_ok=True)

    path = os.path.join(radar_directory, str(day.year), day.strftime("%Y%m%d.tar"))
    if os.path.exists(path):
        with tarfile.open(path) as tar_archive:
            # day_runs = 
            identify_runs(
                tar_archive,
                scans=[
                    file_obj
                    for file_obj in sorted(
                        tar_archive.getmembers()[1:], key=lambda m: m.name
                    )
                ],
                tags=tags,
                dpath=dpath
            )
        # if day_runs is not None:
        #     metadata = []
        #     i = 0
        #     with h5py.File(
        #         os.path.join(
        #             radar_directory, "hdf_archives", day.strftime("%Y%m%d.hdf5")
        #         ),
        #         "w",
        #         libver="latest",
        #     ) as hdf_archive:
        #         for run, periods in day_runs:
        #             avg_value = np.mean(run)
        #             metadata.append(
        #                 {
        #                     "start_datetime": periods[0],
        #                     "end_datetime": periods[1],
        #                     "run_length": run.shape[0],
        #                     "avg_cell_value": avg_value,
        #                     "tags": tags,
        #                 }
        #             )
        #             hdf_archive.create_dataset(
        #                 "{}".format(i),
        #                 chunks=(1, CROP_SIZE, CROP_SIZE),
        #                 shuffle=True,
        #                 fletcher32=True,
        #                 compression="gzip",
        #                 compression_opts=9,
        #                 data=run,
        #             )
        #             i += 1
        #         hdf_archive.flush()
        #     return metadata
