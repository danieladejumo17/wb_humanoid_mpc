#!/usr/bin/env bash
#
# Pin Pinocchio (and its hpp-fcl/eigenpy deps) to a version compatible with this
# codebase.
#
# Why: this workspace targets Pinocchio 2.x (it uses e.g. pinocchio::Frame::parent).
# The default ROS Jazzy apt repo ships Pinocchio 4.0.0, which reorganized headers,
# added joint types (breaking boost::variant/mpl limits) and removed the old API.
# We therefore install Pinocchio 2.6.21 from a pinned 2024 ROS Jazzy snapshot.
#
# Note: system packages are not persistent in every environment. Re-run this
# script after an environment reset if the build starts failing with Pinocchio 4.0
# errors (missing pinocchio/multibody/model.hpp, boost mpl "25 arguments", etc.).
#
# Usage:  sudo bash scripts/setup_pinocchio.sh
set -euo pipefail

SNAPSHOT_DATE="2024-11-21"
PINOCCHIO_VERSION="2.6.21-3noble.20240906.082101"
HPP_FCL_VERSION="2.4.5-1noble.20240906.080357"
EIGENPY_VERSION="3.8.2-1noble.20240906.071821"
# Full fingerprint of the ROS snapshot builder key (short id AD19BAB3CBF125EA).
SNAPSHOT_KEY_FPR="4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA"
KEYRING="/usr/share/keyrings/ros2-snapshot-keyring.gpg"

echo "[setup_pinocchio] Importing ROS snapshot signing key..."
# Fetch the key over HTTPS and dearmor it, rather than using `gpg --recv-keys`.
# In a minimal docker build sandbox `--recv-keys` fails because it needs dirmngr
# (not installed -> "can't connect to the dirmngr") and a writable GnuPG home
# ("No such file or directory" for /root/.gnupg). curl + --dearmor needs neither.
GNUPGHOME="$(mktemp -d)"
export GNUPGHOME
curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x${SNAPSHOT_KEY_FPR}" \
  | gpg --dearmor > "${KEYRING}"
rm -rf "${GNUPGHOME}"

echo "[setup_pinocchio] Adding pinned snapshot apt source (${SNAPSHOT_DATE})..."
cat > /etc/apt/sources.list.d/ros2-snapshot-${SNAPSHOT_DATE}.sources <<EOF
Types: deb
URIs: http://snapshots.ros.org/jazzy/${SNAPSHOT_DATE}/ubuntu
Suites: noble
Components: main
Signed-By: ${KEYRING}
EOF

# Low priority so it never mass-downgrades unrelated packages; explicit installs
# below override this.
cat > /etc/apt/preferences.d/ros2-snapshot-pin <<'EOF'
Package: *
Pin: origin snapshots.ros.org
Pin-Priority: 100
EOF

echo "[setup_pinocchio] Updating apt and installing Pinocchio 2.6.21..."
apt-get update
apt-get install -y --allow-downgrades \
  "ros-jazzy-pinocchio=${PINOCCHIO_VERSION}" \
  "ros-jazzy-hpp-fcl=${HPP_FCL_VERSION}" \
  "ros-jazzy-eigenpy=${EIGENPY_VERSION}"

echo "[setup_pinocchio] Installed Pinocchio version:"
grep -E 'define PINOCCHIO_(MAJOR|MINOR|PATCH)_VERSION' \
  /opt/ros/jazzy/include/pinocchio/config.hpp
