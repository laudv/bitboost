import os
import setuptools
import distutils
import subprocess

from setuptools.command.install import install
from setuptools.command.install_lib import install_lib

PYTHON_DIR = os.path.dirname(__file__)
CARGO_DIR = os.path.join(PYTHON_DIR, "..")
CONFIG_CSV = "bitboost_config.gen.csv"

# Useful examples and docs:
#
# https://github.com/microsoft/LightGBM/blob/master/python-package/setup.py
# https://github.com/wannesm/dtaidistance/blob/master/setup.py
# https://jichu4n.com/posts/how-to-add-custom-build-steps-and-commands-to-setuppy/
# https://docs.python.org/3/distutils/extending.html

class BuildRustCommand(distutils.cmd.Command):
    description = "Build the Rust source files using stable Cargo"
    user_options = [
        # (long option, short option, description)
        ("cargo-bin=", None, "Path to cargo binary"),
        ("release=", None, "Whether to build in release mode")
    ]

    def initialize_options(self):
        self.cargo_bin = "/usr/bin/where-is-cargo"

        # UNIX
        path = os.environ["PATH"]
        path_elems = path.split(":")
        path_elems.reverse()
        for p in path_elems:
            for b in os.listdir(p):
                if b == "cargo":
                    print("CARGO-BIN MATCH:", os.path.join(p, b))
                    self.cargo_bin = os.path.join(p, b)

        print("Set cargo-bin to", self.cargo_bin)

        self.release = True

    def finalize_options(self):
        assert os.path.exists(self.cargo_bin), "Invalid cargo-bin. Is Rust installed?"

        relstr = str(self.release).lower()
        assert relstr in ["y", "yes", "t", "true", "no", "n", "false", "f"], "Invalid release option"

        self.release = relstr in ["y", "yes", "t", "true"]

    def run(self):
        # just check the version
        subprocess.call([self.cargo_bin, "--version"])

        # compile using cargo
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-C target-cpu=native"
        command = [self.cargo_bin, "build"]
        if self.release:
            command.append("--release")
        print("running", " ".join(command))
        subprocess.check_call(command, env=env)

        # generate settings csv
        config_csv = os.path.join(PYTHON_DIR, CONFIG_CSV)
        command = [self.cargo_bin, "run", "--bin", "bitboost_gen_config_csv", config_csv]
        print("running", " ".join(command))
        subprocess.check_call(command, env=env)

class CustomInstall(install):
    """
    Also build the Rust code using Cargo. Not sure whether this should go in a
    custom Build command...
    """
    def run(self):
        self.run_command("build_rust")
        # https://github.com/pypa/setuptools/issues/456#issuecomment-202922033
        #super().run() # does not install dependencies because it runs in backward-compatibile mode :-(
        super().do_egg_install() # call this directly, as it should in super().run()

class CustomInstallLib(install_lib):
    """
    Copy the BitBoost .so to the installation folder.
    """
    def install(self):
        outfiles = super().install()

        # .so shared library
        src_so = self._find_lib()
        dst_so = os.path.join(self.install_dir, "bitboost")
        dst, _ = self.copy_file(src_so, dst_so)
        outfiles.append(dst)

        # config as csv (produced by cargo run --bin bitboost_gen_config_csv)
        src_csv = self._find_config_csv()
        dst_csv = os.path.join(self.install_dir, "bitboost")
        dst, _ = self.copy_file(src_csv, dst_csv)
        outfiles.append(dst)

        return outfiles

    def _find_lib(self):
        release_so = os.path.join(CARGO_DIR, "target/release/libbitboost.so")
        assert os.path.exists(release_so), "where is my so?"
        return release_so

    def _find_config_csv(self):
        config_csv = os.path.join(PYTHON_DIR, CONFIG_CSV)
        assert os.path.exists(config_csv), "where is my config_csv?"
        return config_csv

with open(os.path.join(CARGO_DIR, "README.md")) as f:
    LONG_DESCRIPTION = f.read()

with open(os.path.join(CARGO_DIR, "VERSION")) as f:
    VERSION = f.read().strip()

with open(os.path.join(PYTHON_DIR, "requirements.txt")) as f:
    install_requires = [pkg for pkg in f.read().splitlines() if len(pkg) > 0]

setuptools.setup(
    name="bitboost",
    version=VERSION,
    author="Laurens Devos",
    maintainer="Laurens Devos",
    install_requires=install_requires,
    description="BitBoost: fast gradient boosting with categorical data",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/laudv/bitboost",
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    license="???", # to be decided
    cmdclass={
        'build_rust': BuildRustCommand,
        "install": CustomInstall,
        "install_lib": CustomInstallLib,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        #"License :: ???",
    ]
)
