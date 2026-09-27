"""Can a reviewer install and import this? Offline, no network, no cost.

    python3 check_buildable.py

Checks the BUILD path rather than the analysis path. Two blockers were found here
by hand, both of which made the repository unusable for anyone starting from a
clean checkout, and neither of which any existing suite would have noticed:

  1. requirements.txt used upstream's 'accelerate#==0.23.0' form. pip needs
     whitespace before '#' to treat it as an inline comment; without it the whole
     token is parsed as a requirement name and pip fails on LINE 1, installing
     nothing at all.
  2. src/defense.py evaluates stopwords.words('english') at module level, so the
     nltk stopwords corpus is needed AT IMPORT. Only the spacy model download was
     documented, so the first run ended in an undocumented LookupError.

Both are invisible to a developer whose environment is already built, which is
why they survived. This encodes them.
"""
import re
import sys
import os

FAIL = []


def check(cond, label, detail=''):
    print(f"  [{'  ok  ' if cond else ' FAIL '}] {label}")
    if detail and not cond:
        print(f"            {detail}")
    if not cond:
        FAIL.append(label)


print('=== requirements.txt is parseable by pip ===')
req = open('requirements.txt').read()
lines = [l for l in req.splitlines() if l.strip() and not l.strip().startswith('#')]
bad = [l for l in lines if re.search(r'[A-Za-z0-9_.\-]#', l)]
check(not bad, 'no requirement has "#" without preceding whitespace',
      'pip parses the whole token as a name and fails on that line: ' + '; '.join(bad))

# pip's own parser, if available, is the authority
try:
    from pip._vendor.packaging.requirements import Requirement
    unparseable = []
    for l in lines:
        spec = l.split(' #')[0].split('\t#')[0].strip()
        if not spec:
            continue
        try:
            Requirement(spec)
        except Exception as e:
            unparseable.append(f'{spec!r} ({type(e).__name__})')
    check(not unparseable, 'every requirement parses under packaging.Requirement',
          '; '.join(unparseable))
except ImportError:
    print('  [ skip ] packaging.Requirement unavailable; relied on the "#" rule above')

print('\n=== non-pip downloads are documented ===')
docs = req + '\n' + (open('ARTIFACT.md').read() if os.path.exists('ARTIFACT.md') else '')
check('spacy download en_core_web_sm' in docs,
      'the spacy model download is documented')
check(re.search(r"nltk.*download\(\s*['\"]stopwords", docs) is not None,
      'the nltk stopwords download is documented',
      "src/defense.py evaluates stopwords.words('english') at module level, so "
      'this is required at import, not at first use')

print('\n=== module-level requirements of src/defense.py are all covered ===')
src = open('src/defense.py').read()
head = src[:src.index('class ') if 'class ' in src else len(src)]
top = set(re.findall(r'^\s*(?:import|from)\s+([A-Za-z_][\w]*)', head, re.M))
declared = {l.split()[0].split('#')[0].split('=')[0].split('[')[0].strip().lower()
            for l in lines}
ALIAS = {'sklearn': 'scikit-learn', 'yaml': 'pyyaml'}
STDLIB = {'logging', 'collections', 'itertools', 'copy', 'os', 'json', 'random',
          'dataclasses', 'typing', 're', 'sys', 'string', 'math', 'time'}
missing = sorted(m for m in top - STDLIB
                 if ALIAS.get(m, m).lower() not in declared and not m.startswith('_'))
check(not missing, 'every third-party module imported at module level is in '
                   'requirements.txt', f'not declared: {missing}')

print('\n' + '=' * 60)
if FAIL:
    print(f'{len(FAIL)} BUILD CHECK(S) FAILED -- a reviewer could not build this')
    sys.exit(1)
print('build path is sound: requirements parse, downloads documented, '
      'imports declared')
