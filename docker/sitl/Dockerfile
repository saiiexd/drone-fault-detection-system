# PX4 SITL (software-in-the-loop) with Gazebo Harmonic, headless.
#
# The official px4io/px4-dev image has the PX4 build toolchain but no
# simulator yet (a px4-sim image is planned upstream), so Gazebo is added
# here with PX4's own setup script, and PX4 is compiled for SITL at build time.
#
#   docker build -t aeropulse/px4-sitl:v1.17.0 docker/sitl
#
ARG PX4_VERSION=v1.17.0
FROM px4io/px4-dev:${PX4_VERSION}
ARG PX4_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends sudo git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch ${PX4_VERSION} --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git /px4

# Simulation dependencies (Gazebo Harmonic on Ubuntu 24.04) via PX4's script.
RUN cd /px4 && bash Tools/setup/ubuntu.sh --no-nuttx \
    && rm -rf /var/lib/apt/lists/*

# Build SITL once, at image build time.
RUN cd /px4 && make px4_sitl_default \
    && chmod -R a+rwX /px4/build

COPY run_sitl.sh /usr/local/bin/run_sitl.sh
RUN chmod +x /usr/local/bin/run_sitl.sh

# Gazebo and PX4 write caches and state under HOME; keep it writable for any uid.
# gz-transport discovery uses multicast, which fails on some host interfaces
# under --network host; pin it to loopback (PX4 and Gazebo share the container).
ENV HOME=/tmp HEADLESS=1 GZ_IP=127.0.0.1

# Demo world with scenery (worlds/town.sdf). Its Gazebo Fuel models are
# downloaded here, so the world loads offline and the same way every time.
# Some models reference their textures by bare file name although the files
# sit in materials/textures; copying them next to the meshes lets Gazebo find
# them (otherwise those buildings render black).
ENV GZ_FUEL_CACHE_PATH=/opt/gz-fuel
COPY worlds/town.sdf /px4/Tools/simulation/gz/worlds/town.sdf
RUN mkdir -p "$GZ_FUEL_CACHE_PATH" \
    && grep -o 'https://fuel[^<]*' /px4/Tools/simulation/gz/worlds/town.sdf | sort -u \
    | while read -r url; do gz fuel download -u "$url" || exit 1; done \
    && find "$GZ_FUEL_CACHE_PATH" -type d -path '*/materials/textures' | while read -r t; do \
         [ -d "${t%/materials/textures}/meshes" ] && cp -n "$t"/* "${t%/materials/textures}/meshes/"; \
       done; \
    chmod -R a+rX "$GZ_FUEL_CACHE_PATH" \
    && rm -rf "$HOME/.gz"   # root-owned leftovers would block Gazebo, which runs as the host user

# Window layout for watching a flight (scenarios.py --gui).
COPY viewer.gui.config /opt/aeropulse/viewer.gui.config

WORKDIR /px4
ENTRYPOINT ["/usr/local/bin/run_sitl.sh"]
