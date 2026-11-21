import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from quality import assess,reference_digest
from export import structural_audit

AYAH={'surah':1,'ayah':1,'text':'بسم الله الرحمن الرحيم','asr_text':'بسم الله الرحمن الرحيم'}
SOURCE='a'*64
REF=reference_digest([AYAH])

def observation(shift=0):
    words=[{'word':w,'start_ms':1000+i*500,'end_ms':1400+i*500,'probability':.95} for i,w in enumerate(AYAH['text'].split())]
    return dict(window_start_ms=shift,window_end_ms=5000+shift,source_sha256=SOURCE,
                reference_sha256=REF,model_sha256='b'*64,model_name='small',words=words)

class Ref:
    def track_ayahs(self,*args):return [AYAH]

class AcceptanceTests(unittest.TestCase):
    def test_supported_repeat(self):self.assertEqual(assess(AYAH,[observation(),observation(200)],SOURCE,REF),[])
    def test_duplicate_window_rejected(self):self.assertIn('duplicate_observation_window',assess(AYAH,[observation(),observation()],SOURCE,REF))
    def test_changed_source_rejected(self):self.assertIn('evidence_source_mismatch',assess(AYAH,[observation(),observation(200)],'c'*64,REF))
    def test_word_coverage_required(self):
        o=observation(200);o['words'].pop()
        self.assertIn('reference_word_coverage_mismatch',assess(AYAH,[observation(),o],SOURCE,REF))
    def test_zero_word_rejected(self):
        o=observation(200);o['words'][0]['end_ms']=1000
        self.assertIn('instant_or_overlapping_word',assess(AYAH,[observation(),o],SOURCE,REF))
    def test_unstable_last_word(self):
        o=observation(200);o['words'][-1]['end_ms']+=200
        self.assertIn('unstable_endpoint',assess(AYAH,[observation(),o],SOURCE,REF))
    def test_missing_acoustic_probability(self):
        o=observation(200);o['words'][0]['probability']=None
        self.assertIn('word_probability_missing',assess(AYAH,[observation(),o],SOURCE,REF))
    def test_low_probability_not_approved(self):
        o=observation(200);o['words'][0]['probability']=.01
        self.assertIn('insufficient_acoustic_support',assess(AYAH,[observation(),o],SOURCE,REF))
    def test_large_requires_free_identity(self):
        o=observation(200);o['model_name']='large-v3'
        self.assertIn('large_free_identity_not_verified',assess(AYAH,[observation(),o],SOURCE,REF))
    def track(self):return {'01.mp3':{'unit_number':1,'duration_ms':10000,'verses':[dict(AYAH,num=1,start_ms=1000,end_ms=4000)]}}
    def test_one_ms_export_rejected(self):
        t=self.track();t['01.mp3']['verses'][0]['end_ms']=1001
        self.assertTrue(structural_audit(t,Ref(),'juz'))
    def test_flag_cannot_be_cleared_by_pending_list(self):
        t=self.track();t['01.mp3']['verses'][0]['suspected_boundary']=True
        self.assertTrue(structural_audit(t,Ref(),'juz'))
    def test_wrong_text_export_rejected(self):
        t=self.track();t['01.mp3']['verses'][0]['text']='wrong'
        self.assertTrue(structural_audit(t,Ref(),'juz'))
    def test_supported_structure(self):self.assertEqual(structural_audit(self.track(),Ref(),'juz'),[])
    def test_short_surah_does_not_repeat_the_same_full_file_window(self):
        from verified_align import groups_from_primary
        units={str(n):{'words':[{'start_ms':100+(n-1)*800,'end_ms':500+(n-1)*800}]} for n in range(1,8)}
        groups=groups_from_primary({'units':units},7,6000)
        for g in groups:
            self.assertNotIn('error',g)
            self.assertNotEqual((g['begin'],g['end']),(max(0,g['begin']-310),min(6000,g['end']+310)))

if __name__=='__main__':unittest.main()
