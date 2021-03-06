PHOEBE 2.0 RELEASE NOTES
------------------------

Hello and thank you for your interest in PHOEBE 2.0! PHOEBE is a binary star modeling code, but version 2.0 also supports the modeling of single rotating stars.


INTRODUCTION
------------

PHOEBE stands for PHysics Of Eclipsing BinariEs. PHOEBE is pronounced [fee-bee](https://www.merriam-webster.com/dictionary/phoebe?pronunciation&lang=en_us&file=phoebe01.wav).

PHOEBE 2.0 is a rewrite of the original PHOEBE code. For most up-to-date information please refer to the PHOEBE project webpage: [http://phoebe-project.org](http://phoebe-project.org)

PHOEBE 2.0 is described by the release paper published in the Astrophysical Journal Supplement, [Prša et al. (2016, ApJS 227, 29)](http://adsabs.harvard.edu/abs/2016ApJS..227...29P).

PHOEBE 2.0 is released under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html).


The source code is available for download from the [PHOEBE project homepage](http://phoebe-project.org) and from [github](https://github.com/phoebe-project/phoebe2).

The development of PHOEBE 2.0 is funded in part by the [NSF grant #1517474](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1517474).


DOWNLOAD AND INSTALLATION
-------------------------

The easiest way to download and install PHOEBE 2.0 is by using pip:

    pip install phoebe

To install it site-wide, prefix the `pip` command with `sudo` or run it as root.

To download the PHOEBE 2.0 source code, use git:

    git clone https://github.com/phoebe-project/phoebe2.git

To install PHOEBE 2.0 from the source locally, go to the `phoebe2/` directory and issue:

    python setup.py build
    python setup.py install --user

To install PHOEBE 2.0 from the source site-wide, go to the `phoebe2/` directory and issue:

    python setup.py build
    sudo python setup.py install

For further details on pre-requisites and minimal versions of python consult the PHOEBE webpage.


GETTING STARTED
---------------

PHOEBE 2.0 has a steep learning curve. There is no graphical front-end as of yet; the front-end is now written in python. To start PHOEBE, issue:

    python
    >>> import phoebe
    >>>

To understand how to use PHOEBE, please consult the [tutorials, scripts and manuals](http://phoebe-project.org/docs/2.0/#Tutorials) hosted on the PHOEBE webpage.


CHANGELOG
----------

### 2.0.4 - vgamma temporary bugfix
* The definition of vgamma in 2.0.* is in the direction of positive z rather than positive RV.  For the sake of maintaining backwards-compatibility, this will remain unchanged for 2.0.* releases but will be fixed in the 2.1 release to be in the direction of positive RV.  Until then, this bugfix handles converting to and from PHOEBE legacy correctly so that running the PHOEBE 2 and legacy backends gives consistent results.

### 2.0.3 - t0_supconj/t0_perpass hotfix
* Fixed constraint that defines the relation between t0_perpass and t0_supconj.
* Implement new 't0_ref' parameter which corresponds to legacy's 'HJD0'.
* Phasing now accepts t0='t0_supconj', 't0_perpass', 't0_ref', or a float representing the zero-point.  The 'phshift' parameter will still be supported until 2.1, at which point it will be removed.
* Inclination parameter ('incl') is now limited to the [0-180] range to maintain conventions on superior conjunction and ascending/descending nodes.
* Fixed error message in ldint.
* Fixed the ability for multiple stars to be attached to the same component.
* Raise an error if attempting to attach spots to an unsupported component.  Note: spots are currently not supported for contact systems.

### 2.0.2 - spots hotfix

* If using spots, it is important that you use 2.0.2 or later as there were several important bug fixes in this release.
* 'colon' parameter for spots has been renamed to 'long' (as its not actually colongitude).  For 2.0.X releases, the 'colon' parameter will remain as a constrained parameter to avoid breaking any existing scripts, but will be removed with the 2.1.0 release.
* Features (including spots) have been fixed to correctly save and load to file.
* Corotation of spots is now enabled: if the 'syncpar' parameter is not unity, the spots will correctly corotate with the star.  The location of the spot (defined by 'colat' and 'long' parameters) is defined such that the long=0 points to the companion star at t0.  That coordinate system then rotates with the star according to 'syncpar'.

### 2.0.1 - ptfarea/pbspan hotfix

* Definition of flux and luminosity now use ptfarea instead of pbspan.  In the bolometric case, these give the same quantity. This discrepancy was absorbed entirely by pblum scaling, so relative fluxes should not be affected, but the underlying absolute luminosities were incorrect for passbands (non-bolometric).  In addition to under-the-hood changes, the exposed mesh column for 'pbspan' is now removed and replaced with 'ptfarea', but as this is not yet a documented column, should not cause backwards-compatibility issues.  

### 2.0.0

* PHOEBE 2.0 is not backwards compatible with PHOEBE 2.0-beta (although the interface has not changed appreciably) or with PHOEBE 2.0-alpha (substantial rewrite). Going forward with incremental releases, every effort will be put into backwards compatibility. The changes and important considerations of the new version will be detailed in the ChangeLog.

* If upgrading from PHOEBE 2.0-beta or PHOEBE 2.0-alpha, it is necessary to do a clean re-install (clear your build and installation directories), as the passband file format has changed and will not automatically reset unless these directories are manually cleared.  Contact us with any problems.


QUESTIONS? SUGGESTIONS? CONCERNS?
---------------------------------

Contact us! Issues and feature requests should be submitted directly through GitHub's issue tracker. Two mailing lists are dedicated for discussion, either on user level ([phoebe-discuss@lists.sourceforge.net](mailto:phoebe-discuss@lists.sourceforge.net)) or on the developer level ([phoebe-devel@lists.sourceforge.net](mailto:phoebe-devel@lists.sourceforge.net)). We are eager to hear from you, so do not hesitate to contact us!
