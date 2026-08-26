
import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot a single ceilometer netCDF file.")
    parser.add_argument("filename")
    parser.add_argument("vmin", nargs="?", type=float, default=0.0)
    parser.add_argument("vmax", nargs="?", type=float, default=1e6)
    args = parser.parse_args()

    if not args.filename:
        parser.error("a filename must be provided")

    import matplotlib.pyplot as plt
    import xarray as xr

    plt.ion()
    x = xr.load_dataset(args.filename)

    plt.clf()
    plt.pcolormesh(x.beta_raw.time, x.beta_raw.range, x.beta_raw.T, vmin=args.vmin, vmax=args.vmax)
    plt.plot(x.cbh.time, x.cbh, "or")
    plt.show(block=True)


if __name__ == "__main__":
    main()
