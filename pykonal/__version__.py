__major_version__    = 0
__minor_version__    = 2
__minor_subversion__ = 3
__release__          = "a"
__patch__            = 0
__version_number__   = ".".join(
    (
        str(__major_version__),
        str(__minor_version__),
        str(__minor_subversion__)
    )
)
__version__          = f"{__version_number__}{__release__}{__patch__}"
