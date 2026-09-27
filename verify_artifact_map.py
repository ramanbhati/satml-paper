import re, os, subprocess, sys
doc = open('ARTIFACT.md').read()
bad = []

# 1. every .py / .sh / .md filename mentioned must exist and be tracked
tracked = set(subprocess.check_output(['git','ls-files'], text=True).split())
for f in sorted(set(re.findall(r'`([A-Za-z0-9_./-]+\.(?:py|sh|md|json|txt))`', doc))):
    cand = f.lstrip('./')
    if cand in tracked or os.path.exists(cand): continue
    if cand in ('data/jailbreak_prompts.json','requirements.txt'):  # intentionally absent / present
        if os.path.exists(cand) or cand=='data/jailbreak_prompts.json': continue
    bad.append(f'file mentioned but not found: {f}')

# 2. every results CSV named (non-glob, non-template) must exist
for c in sorted(set(re.findall(r'`(results/[a-z_0-9.]+\.csv)`', doc))):
    if not os.path.exists(c): bad.append(f'CSV mentioned but missing: {c}')
# CSVs the doc explicitly flags as not shipped are exempt
_exempt = {'clean.csv', 'lown_opennq.csv'}
for c in sorted(set(re.findall(r'`([a-z_0-9]+\.csv)`', doc))):
    if c in _exempt: continue
    if not os.path.exists(os.path.join('results', c)):
        bad.append(f'CSV mentioned but missing: results/{c}')
for c in sorted(_exempt):
    if c not in doc: bad.append(f'exempt CSV no longer disclosed as absent: {c}')

# 3. every generated asset named must exist in the paper's generated/, or in the
# local ./generated/ that make_paper_assets.py writes when the paper tree is
# absent (a released copy of this repository).
G = '../../../paper-satml2027/generated'
if not os.path.isdir(G):
    G = 'generated'
if not os.path.isdir(G):
    sys.exit('verify_artifact_map.py: no generated/ directory; run '
             'python3 make_paper_assets.py first')
on_disk = {f[:-4] for f in os.listdir(G) if f.endswith('.tex')}
named = set(re.findall(r'`((?:tab|fig)_[a-z]+|nobs|certify_facts|theirs_facts)`', doc))
for a in sorted(named - on_disk): bad.append(f'asset named but not generated: {a}')
missing_from_doc = on_disk - named
if missing_from_doc: bad.append(f'generated assets NOT documented: {sorted(missing_from_doc)}')
print(f'  assets: {len(named)} named, {len(on_disk)} on disk')

# 4. the upstream commit must exist and src/ paths must be unchanged since it
# The upstream tree is the repository's first commit (HEAD~1), which records
# upstream's own hash, 9bc35b2, in its message. 9bc35b2 itself is not an object
# in this repository.
UP = 'HEAD~1'
subprocess.check_call(['git','cat-file','-e',UP], stdout=subprocess.DEVNULL)
up = subprocess.check_output(['git','ls-tree','-r','--name-only',UP,'src/'], text=True).split()
now = subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','src/'], text=True).split()
if set(up) != set(now): bad.append(f'src/ paths changed since upstream: {set(up)^set(now)}')
print(f'  src/ paths identical to upstream: {set(up)==set(now)} ({len(now)} files)')

# 5. the seven "modified" files must really differ from upstream
for f in ['src/defense.py','src/models.py','src/attack.py','main.py','src/helper.py',
          'src/prompt_template.py','src/dataset_utils.py']:
    rc = subprocess.call(['git','diff','--quiet',UP,'HEAD','--',f])
    if rc == 0: bad.append(f'claimed modified but identical to upstream: {f}')
print('  all seven claimed-modified files do differ from upstream')

# 6. anonymity.
#
# The patterns are READ FROM A FILE, never written here. Hard-coding author and
# institution names as string literals put them in a file that ships with the
# anonymous artifact, so grepping the release for the very names this check
# exists to exclude would have found them. The pattern file is untracked; when
# it is absent this check reports that it did not run rather than passing
# quietly, because a silent pass is indistinguishable from a clean result.
_PATFILE = '.anonymity_patterns'
if os.path.exists(_PATFILE):
    _pats = [l.strip() for l in open(_PATFILE) if l.strip() and not l.startswith('#')]
    for pat in _pats:
        if re.search(re.escape(pat), doc, re.I):
            bad.append(f'de-anonymising string in ARTIFACT.md matching {_PATFILE} entry #{_pats.index(pat)+1}')
    print(f'  anonymity: {len(_pats)} patterns checked, none present')
else:
    print(f'  anonymity: NOT CHECKED -- {_PATFILE} absent.')
    print(f'             Create it (one pattern per line, untracked) to enable.')

if bad:
    print('\nFAILURES:'); [print('  !!', b) for b in bad]; sys.exit(1)
print('\n  ARTIFACT.md verified: every path, CSV, asset and claim checks out')
