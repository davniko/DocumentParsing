from scripts.analysis.r31_boundary_quality import boundary_readback


def example(texts, name="Long Maritime Supplies", address="12 New Road"):
    regions = list(range(len(texts)))
    rendered = {
        'target':{'parties':{'shipper':{'name':name,'addressLine':address}}},
        'proof':[{'owner':'party@0:10','spans':[{'page':0,'region':i} for i in regions]}],
        'geometry':{'pages':[{'page':0,'blocks':[{'sourceRegions':regions,
            'physicalFragments':[{'text':t,'box':[0,i*20,400,i*20+15]} for i,t in enumerate(texts)]}]}]},
    }
    candidate={'proof':{'edits':[{'key':'party','byteStart':0,'byteEnd':10,
        'targetPaths':['documentPatch.parties.shipper.name','documentPatch.parties.shipper.addressLine']} ]}}
    return rendered,candidate


def test_continuation_measured_on_actual_final_lines():
    r,c=example(['Long Maritime','Supplies.12 New Road'])
    result=boundary_readback(r,c,{})[0]
    assert [m['kind'] for m in result['occurrences']]==['continuation_shared']
    assert result['occurrences'][0]['boundary']['text']=='Supplies.12 New Road'


def test_short_legal_suffix_is_not_missed():
    r,c=example(['Long Maritime','LLC 12 New Road'], name='Long Maritime LLC')
    assert boundary_readback(r,c,{})[0]['occurrences'][0]['kind']=='continuation_shared'


def test_whole_name_is_not_misreported_as_continuation():
    r,c=example(['Long Maritime Supplies 12 New Road'])
    assert boundary_readback(r,c,{})[0]['occurrences'][0]['kind']=='whole_name_shared'


def test_separate_frames_are_not_hard_examples():
    r,c=example(['Long Maritime Supplies','12 New Road'])
    assert boundary_readback(r,c,{})[0]['occurrences'][0]['kind']=='separate'


def test_foreign_roles_cannot_supply_shared_boundary():
    r,c=example(['Long Maritime Supplies','12 New Road'])
    r['geometry']['pages'][0]['blocks'].append({'sourceRegions':[9],
        'physicalFragments':[{'text':'Long Maritime Supplies 12 New Road','box':[600,0,990,15]}]})
    assert [m['kind'] for m in boundary_readback(r,c,{})[0]['occurrences']]==['separate']


def test_missing_text_stays_unresolved():
    r,c=example(['Long Maritime Supplies 21 Old Road'])
    assert boundary_readback(r,c,{})[0]['status']=='needs_review'
