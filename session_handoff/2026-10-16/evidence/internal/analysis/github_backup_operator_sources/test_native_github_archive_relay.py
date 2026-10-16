"""Lightweight tests for capped buffers and private archive integrity."""
import hashlib,io,unittest
from unittest.mock import patch
import native_github_archive_relay as relay_module
from native_github_archive_relay import BoundedReader,validate_spec,verify_asset

class RelayTests(unittest.TestCase):
    def setUp(self):
        self.payload=b'x'*(2*1024*1024+11)
        self.spec={'name':'N15R18_s0_checkpoint-100.tar.gz',
            'path':'/content/native_github_staging/N15R18_s0_checkpoint-100.tar.gz',
            'bytes':len(self.payload),'sha256':hashlib.sha256(self.payload).hexdigest()}

    def test_reader_caps_even_unbounded_requests_and_tracks_digest(self):
        reader=BoundedReader(io.BytesIO(self.payload),len(self.payload),buffer_bytes=1024*1024)
        parts=[]
        while data:=reader.read(-1):parts.append(data)
        self.assertEqual(b''.join(parts),self.payload)
        self.assertLessEqual(reader.max_chunk,1024*1024)
        self.assertEqual(reader.sha256.hexdigest(),self.spec['sha256'])

    def test_reader_caps_large_requested_chunks(self):
        reader=BoundedReader(io.BytesIO(self.payload),len(self.payload),buffer_bytes=8192)
        self.assertEqual(len(reader.read(16*1024*1024)),8192)

    def test_requests_preserves_stream_identity_and_fixed_content_length(self):
        import requests
        reader=BoundedReader(io.BytesIO(self.payload),len(self.payload))
        request=requests.Request('POST','https://uploads.github.com/example',data=reader,
            headers={'Content-Length':str(len(self.payload))}).prepare()
        self.assertIs(request.body,reader)
        self.assertEqual(request.headers['Content-Length'],str(len(self.payload)))
        self.assertNotIn('Transfer-Encoding',request.headers)

    def test_reader_rejects_early_end_or_extra_source_bytes(self):
        for data,length in [(b'x',2),(b'xxx',2)]:
            reader=BoundedReader(io.BytesIO(data),length,buffer_bytes=8)
            with self.assertRaises(ValueError):
                while reader.read():pass

    def test_spec_restricts_archive_names_paths_and_digest(self):
        self.assertEqual(validate_spec(self.spec),self.spec)
        for patch in [{'name':'other-repo.zip'},{'path':'/content/../secret.json'},
                      {'sha256':'bad'},{'bytes':0},{'bytes':3*1024**3}]:
            with self.assertRaises(ValueError):validate_spec({**self.spec,**patch})

    def test_asset_requires_uploaded_state_matching_size_and_server_digest(self):
        asset={'name':self.spec['name'],'size':self.spec['bytes'],'state':'uploaded',
            'digest':'sha256:'+self.spec['sha256']}
        self.assertTrue(verify_asset(asset,self.spec))
        for patch in [{'name':'wrong'},{'size':1},{'state':'starter'},
                      {'digest':'sha256:'+'0'*64}]:
            with self.assertRaises(ValueError):verify_asset({**asset,**patch},self.spec)

    def test_reuses_matching_archive_on_second_asset_page(self):
        asset={'id':123,'name':self.spec['name'],'size':self.spec['bytes'],
            'state':'uploaded','digest':'sha256:'+self.spec['sha256']}
        pages=[]
        def metadata(endpoint,method='GET'):
            if endpoint=='repos/owner/archive':return {'private':True,'owner':{'login':'owner'}}
            if endpoint.endswith('/releases/42'):return {'draft':True,'published_at':None}
            if '/assets?' in endpoint:
                pages.append(endpoint)
                return [asset] if 'page=2' in endpoint else [{'name':'unrelated-'+str(i)} for i in range(100)]
            raise AssertionError(endpoint)
        with patch.object(relay_module,'gh_json',side_effect=metadata),patch('requests.get',side_effect=AssertionError('Unexpected upload')):
            receipt=relay_module.relay(self.spec,{'repo':'owner/archive','account_login':'owner','release_id':42})
        self.assertTrue(receipt['reused'])
        self.assertEqual(len(pages),2)

    def test_empty_starter_is_deleted_before_retrying_upload(self):
        empty={'id':123,'name':self.spec['name'],'size':0,'state':'starter'}
        deleted=[]
        class UploadReached(Exception):pass
        def metadata(endpoint,method='GET'):
            if method=='DELETE':deleted.append(endpoint);return None
            if endpoint=='repos/owner/archive':return {'private':True,'owner':{'login':'owner'}}
            if endpoint.endswith('/releases/42'):
                return {'draft':True,'published_at':None,'upload_url':'https://uploads.github.com/repos/owner/archive/releases/42/assets{?name,label}'}
            if '/assets?' in endpoint:return [empty]
            raise AssertionError(endpoint)
        from types import SimpleNamespace
        session=SimpleNamespace(url='https://example.colab.dev',token='test-only')
        with patch.object(relay_module,'gh_json',side_effect=metadata),patch('colab_cli.common.State') as state,patch('requests.get',side_effect=UploadReached),patch.object(relay_module.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='test-only')):
            state.return_value.get_session.return_value=session
            with self.assertRaises(UploadReached):
                relay_module.relay(self.spec,{'repo':'owner/archive','account_login':'owner','release_id':42})
        self.assertEqual(deleted,['repos/owner/archive/releases/assets/123'])

    def test_reuse_rechecks_privacy_before_writing_receipt(self):
        asset={'id':123,'name':self.spec['name'],'size':self.spec['bytes'],
            'state':'uploaded','digest':'sha256:'+self.spec['sha256']}
        checks=[]
        def metadata(endpoint,method='GET'):
            if endpoint=='repos/owner/archive':
                checks.append(endpoint)
                return {'private':len(checks)==1,'owner':{'login':'owner'}}
            if endpoint.endswith('/releases/42'):return {'draft':True,'published_at':None}
            if '/assets?' in endpoint:return [asset]
            raise AssertionError(endpoint)
        with patch.object(relay_module,'gh_json',side_effect=metadata):
            with self.assertRaisesRegex(ValueError,'private'):
                relay_module.relay(self.spec,{'repo':'owner/archive','account_login':'owner','release_id':42})

if __name__=='__main__':unittest.main()
