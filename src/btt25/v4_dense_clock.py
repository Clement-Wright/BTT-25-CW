"""Additional trained 20 ms clock. The registered 12-condition source stays byte-identical."""
from .v4_rate import RateSpec, SPECS


def register():
    SPECS["gru20"] = RateSpec(stride=1)


if __name__ == "__main__":
    register()
    from .v4_rate_study import main
    main()
