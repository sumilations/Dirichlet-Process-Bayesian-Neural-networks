import re
import os
import glob

def verify():
    tex_path = 'main_tmlr.tex'
    bib_path = 'main.bib'

    with open(tex_path, 'r', encoding='utf-8') as f:
        tex = f.read()

    print("====================================================")
    print("      TMLR CAMERA-READY AUDIT & VERIFICATION       ")
    print("====================================================")

    # 1. Images
    images = re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', tex)
    print(f"\n[1] Checking {len(images)} figure inclusions in LaTeX...")
    missing_images = []
    for img in images:
        # Check direct path or with extensions
        found = False
        candidates = [img, img + '.pdf', img + '.png', img + '.jpg', img + '.eps']
        for cand in candidates:
            if os.path.exists(cand):
                found = True
                print(f"  [OK] {img} (found: {cand})")
                break
        if not found:
            print(f"  [MISSING] {img}")
            missing_images.append(img)

    # 2. Labels and References
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = set(re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}', tex))
    print(f"\n[2] Checking Labels and References...")
    print(f"  Total labels defined: {len(labels)}")
    print(f"  Total references invoked: {len(refs)}")
    missing_labels = refs - labels
    if missing_labels:
        print(f"  [ERROR] Found missing labels for references: {missing_labels}")
    else:
        print("  [OK] All \\ref and \\eqref have valid matching \\label definitions!")

    # Check for duplicate labels
    all_label_instances = re.findall(r'\\label\{([^}]+)\}', tex)
    duplicates = [l for l in labels if all_label_instances.count(l) > 1]
    if duplicates:
        print(f"  [WARNING] Duplicate labels detected: {duplicates}")
    else:
        print("  [OK] No duplicate labels detected.")

    # 3. Citations
    with open(bib_path, 'r', encoding='utf-8') as f:
        bib = f.read()

    bib_keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+),', bib))
    print(f"\n[3] Checking Bibliography & Citations...")
    print(f"  Total BibTeX entries defined: {len(bib_keys)}")

    cite_matches = re.findall(r'\\(?:cite|citep|citet|citealp|citealt)\*?\{([^}]+)\}', tex)
    used_citations = set()
    for c in cite_matches:
        for k in c.split(','):
            cleaned = k.strip()
            if cleaned:
                used_citations.add(cleaned)

    print(f"  Total citation keys invoked: {len(used_citations)}")
    missing_cites = used_citations - bib_keys
    if missing_cites:
        print(f"  [ERROR] Missing BibTeX entries ({len(missing_cites)}): {missing_cites}")
    else:
        print("  [OK] All citation keys exist in main.bib!")

    # 4. Check for place holders / artifacts
    print(f"\n[4] Scanning for leftover placeholders and markers...")
    placeholders = re.findall(r'(\bTODO\b|\bFIXME\b|\?\?\?|\\fix\b|\\new\b)', tex)
    # Check lines where fix or new are actually invoked (not just \newcommand{\fix})
    fix_invocations = [m.start() for m in re.finditer(r'\\(?:fix|new)\b', tex)]
    # Filter out definitions
    active_fix_new = []
    for line_idx, line in enumerate(tex.splitlines(), 1):
        if re.search(r'\\(?:fix|new)\s*\{', line):
            active_fix_new.append((line_idx, line))

    if active_fix_new:
        print(f"  [WARNING] Active \\fix or \\new found in text:")
        for l_num, l_text in active_fix_new:
            print(f"    Line {l_num}: {l_text}")
    else:
        print("  [OK] No active \\fix or \\new margin notes.")

    other_todos = [p for p in placeholders if p not in ['\\fix', '\\new']]
    if other_todos:
        print(f"  [WARNING] Found markers: {set(other_todos)}")
    else:
        print("  [OK] No TODO / FIXME / ??? markers.")

    # 5. Check Section structure and Table 1
    print(f"\n[5] Checking Document Structure & Key Tables/Figures...")
    has_table_summary = 'tab:empirical_summary' in tex
    print(f"  Consolidated Benchmark Summary Table (tab:empirical_summary): {'PRESENT' if has_table_summary else 'MISSING'}")
    has_deepsea50 = 'DeepSea' in tex and '50' in tex
    print(f"  DeepSea N=50 mentions: {'PRESENT' if has_deepsea50 else 'MISSING'}")
    has_scaling_figure = ('Figure_3_TMLR.pdf' in tex) or ('Figure_2_TMLR.pdf' in tex)
    print(f"  Main DeepSea Performance & Scaling Figure: {'PRESENT' if has_scaling_figure else 'MISSING'}")

    print("\n====================================================")
    print("                 AUDIT COMPLETE                     ")
    print("====================================================")

if __name__ == '__main__':
    verify()
