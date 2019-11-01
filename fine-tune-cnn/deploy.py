#!/usr/bin/env python3

"""Deploy the training to the remote instance with the GPU."""

import argparse
import contextlib
import os
import pathlib
import sys
import textwrap
import uuid

import spur
import spurplus


def main() -> int:
    """Execute the main routine."""
    ##
    # Specify command-line arguments
    ##

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hostname",
        help="Host name of the remote instance",
        required=True)

    parser.add_argument(
        "--user",
        help="Remote user used for the deployment",
        default="devop")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--data_dir",
        help="Root directory where data sets reside; "
             "if not specified, the data is not synced")

    group.add_argument(
        "--data_url",
        help="URL where data sets reside; "
             "if not specified, the data is not downloaded on the remote host")

    parser.add_argument(
        "--dont_install_requirements",
        help="If set, requirements are not installed on the remote machine.",
        action='store_true')

    ##
    # Parse command-line arguments
    ##

    args = parser.parse_args()

    hostname = str(args.hostname)
    remote_user = str(args.user)

    data_dir = None if args.data_dir is None else pathlib.Path(args.data_dir)
    data_url = None if args.data_url is None else str(args.data_url)

    assert ((data_dir is None and data_url is None) or
            ((data_dir is None) ^ (data_url is None)))

    if data_dir is not None and not data_dir.exists():
        raise FileNotFoundError("Data directory does not exist: {}".format(
            data_dir))

    dont_install_requirements = bool(args.dont_install_requirements)

    ##
    # Connect
    ##

    print("Connecting to the remote instance at {}@{} ...".format(
        remote_user, hostname))

    with spurplus.connect_with_retries(
            hostname=hostname,
            username=remote_user,
            missing_host_key=spur.ssh.MissingHostKey.warn) as ssh:
        ##
        # Specify common paths
        ##

        remote_home_dir = pathlib.Path(
            ssh.check_output(['/bin/bash', '-c', 'echo $HOME']).strip())

        remote_mediti_dir = remote_home_dir / "mediti-train"

        script_dir = pathlib.Path(os.path.realpath(__file__)).parent

        ##
        # Install requirements
        ##

        if not dont_install_requirements:
            print("Installing the requirements...")

            ssh.put(
                local_path=script_dir / "requirements-gpu.txt",
                remote_path=remote_mediti_dir / "requirements-gpu.txt")

            install_pth = remote_mediti_dir / 'install.sh'
            ssh.write_text(
                remote_path=install_pth,
                text=textwrap.dedent(
                    '''\
                    #!/bin/bash
                    set -e
                    echo "sudo apt-get install'ing ..."
                    sudo apt-get install -y python3-venv wget unzip
                    
                    echo "Creating the virtual environment ..."
                    python3 -m venv venv
                    source venv/bin/activate
                    
                    echo "Installing the python requirements ..."
                    pip3 install -r requirements-gpu.txt
                    '''))

            ssh.chmod(remote_path=install_pth, mode=0o700)

            ssh.run(command=[install_pth.as_posix()], cwd=remote_mediti_dir)

        ##
        # Sync the data
        ##

        if data_dir is not None:
            print("Syncing the data...")

            assert data_dir is not None
            remote_data_dir = remote_mediti_dir / "data"
            ssh.mkdir(remote_path=remote_data_dir, exist_ok=True)
            ssh.sync_to_remote(
                local_path=data_dir, remote_path=remote_data_dir,
                delete=spurplus.Delete.BEFORE)

        if data_url is not None:
            remote_data_dir = remote_mediti_dir / "data"
            remote_limbo_dir = remote_mediti_dir / "data-limbo.{}".format(
                uuid.uuid4())

            def remove_limbo_dir() -> None:
                """Delete the temporary data limbo directory."""
                if ssh.exists(remote_limbo_dir):
                    print("Removing the limbo directory: {}".format(
                        remote_limbo_dir))
                    assert remote_limbo_dir != pathlib.Path()
                    assert remote_mediti_dir in remote_limbo_dir.parents
                    assert remote_limbo_dir != remote_mediti_dir
                    ssh.run(['rm', '-rf', remote_limbo_dir.as_posix()])

            with contextlib.ExitStack() as exit_stack:
                exit_stack.callback(remove_limbo_dir)

                ssh.mkdir(remote_limbo_dir, exist_ok=True, parents=True)
                ssh.mkdir(remote_data_dir, exist_ok=True, parents=True)

                print("Downloading the data to the remote limbo: {}".format(
                    remote_limbo_dir))

                # yapf: disable
                ssh.run([
                    'wget', data_url,
                    '-o', (remote_limbo_dir / "data.zip").as_posix()])
                # yapf: enable

                # yapf: disable
                ssh.run([
                    'unzip', (remote_limbo_dir / "data.zip").as_posix(),
                    '-d', (remote_data_dir).as_posix()
                ])
                # yapf: enable

        ##
        # Sync the code
        ##

        print("Syncing the code...")

        rel_pths = [
            pathlib.Path("fine_tuneti.py"),
            pathlib.Path("evaluateti.py"),
            pathlib.Path("file_iterator.py"),
            pathlib.Path("specsmod.py"),
        ]

        remote_src_pth = remote_mediti_dir / "src"

        for rel_pth in rel_pths:
            ssh.put(
                local_path=script_dir / rel_pth,
                remote_path=remote_src_pth / rel_pth)

        for rel_pth in [pathlib.Path("fine_tuneti.py"),
                        pathlib.Path("evaluateti.py")]:
            ssh.chmod(remote_path=remote_src_pth / rel_pth, mode=0o700)

        ##
        # Goodbye
        ##

        print("The deployment has finished.")
        print("Execute manually to train and evaluate on: {}@{}".format(
            remote_user, hostname))

    return 0


if __name__ == "__main__":
    sys.exit(main())
