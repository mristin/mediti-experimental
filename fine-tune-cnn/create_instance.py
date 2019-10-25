#!/usr/bin/env python3

"""Create an instance on Google Cloud."""

import subprocess
import time

INSTANCE_NAME = "mediti-train"


def main() -> None:
    """Execute the main routine."""
    ##
    # Create the instance
    ##

    print("Creating the instance {}...".format(INSTANCE_NAME))

    # yapf: disable
    subprocess.check_call([
        'gcloud', 'compute', 'instances', 'create', INSTANCE_NAME,
        '--zone=europe-west1-b',
        '--image-family=tf2-latest-gpu',
        '--image-project=deeplearning-platform-release',
        '--maintenance-policy=TERMINATE',
        '--accelerator', 'type=nvidia-tesla-p100,count=1',
        '--machine-type=n1-highmem-8',
        '--boot-disk-size=200GB',
        '--metadata', 'install-nvidia-driver=True',
        '--preemptible',
        '--verbosity', 'info'])
    # yapf: enable

    ##
    # Authorize the key
    ##

    print("SSHing via gcloud compute to authorize the key...")
    retries = 6
    success = False
    for _ in range(retries):
        retcode = subprocess.call([
            "gcloud", "compute", "ssh", "devop@{}".format(INSTANCE_NAME),
            "--command", "echo oi"])

        if retcode == 0:
            success = True
            break

        time.sleep(30)

    if not success:
        raise RuntimeError(
            "Failed to gcloud compute ssh to the instance: devop@{}".format(
                INSTANCE_NAME))


if __name__ == "__main__":
    main()
