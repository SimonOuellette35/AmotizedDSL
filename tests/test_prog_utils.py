from AmotizedDSL.prog_utils import ProgUtils
import AmotizedDSL.DSL as DSL
import random
import uuid
import numpy as np
from unittest.mock import patch


def test_remove_unused_instructions():
    unused_instrs = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)', 
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)', 
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)', 
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)', 
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)', 
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)', 
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)', 
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)', 
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)', 
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a999-8b81-48f6-b38a-088ca65ed389, 4)', 
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)', 
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)', 
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)', 
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b86666-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)', 
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)'
    ]

    cleaned_prog = ProgUtils.remove_unused_instructions(unused_instrs)

    expected_cleaned_prog = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)',
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)',
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a999-8b81-48f6-b38a-088ca65ed389, 4)',
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)',
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)',
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b86666-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)'
    ]

    assert cleaned_prog == expected_cleaned_prog, "cleaned_prog does not match expected_cleaned_prog"

def test_reassign_invalid_uuids():
    invalid_uuid_refs = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)', 
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)', 
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)', 
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)', 
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)', 
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)', 
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)', 
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)', 
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)', 
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a999-8b81-48f6-b38a-088ca65ed389, 4)', 
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)', 
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)', 
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)', 
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b86666-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)', 
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)'
    ]

    # Set deterministic seeds for all random number generators
    np.random.seed(42)
    random.seed(42)
    
    # Patch np.random.choice to sort list inputs for deterministic selection
    # This ensures that when list(valid_uuids) is passed, it's sorted first
    original_choice = np.random.choice
    def deterministic_choice(a, **kwargs):
        # Sort list inputs to ensure deterministic order (handles list(valid_uuids) case)
        if isinstance(a, list):
            a = sorted(a)
        elif isinstance(a, (set, frozenset)):
            a = sorted(a)
        return original_choice(a, **kwargs)
    
    with patch('AmotizedDSL.prog_utils.np.random.choice', side_effect=deterministic_choice):
        fixed_uuid_refs = ProgUtils.reassign_invalid_uuids(invalid_uuid_refs, 7)

    expected_uuid_refs = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)',
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)',
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 4)',
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)',
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)',
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(371ecd7b-27cd-4130-8722-9389571aa876, 5be6128e-18c2-4797-a142-ea7d17be3111)',
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)'
    ]

    assert fixed_uuid_refs == expected_uuid_refs, "fixed_uuid_refs does not match expected_uuid_refs"
    
    # Assert that the only two rows that are different between expected_uuid_refs and invalid_uuid_refs are rows 9 and 13
    different_rows = [i for i in range(len(fixed_uuid_refs)) if fixed_uuid_refs[i] != invalid_uuid_refs[i]]
    assert set(different_rows) == {9, 13}, f"Expected only rows 9 and 13 to differ, but found differences at rows: {different_rows}"


def test_map_refIDs_to_uuids():
    # Set deterministic seed for UUID generation
    random.seed(42)
    
    # Create a deterministic UUID generator
    def deterministic_uuid4():
        # Generate UUID using random module (which respects seed)
        return uuid.UUID(int=random.getrandbits(128), version=4)
    
    # Patch uuid.uuid4 to use deterministic generator
    # Since uuid is imported inside map_refIDs_to_uuids, we patch the standard library uuid module
    with patch('uuid.uuid4', side_effect=deterministic_uuid4):
        prog_example = [
            'get_objects(N+0)',
            'get_bg(N+0)',
            'del(N+0)',
            'neighbours8(N+0)',
            'neighbours4(N+0)',
            'set_difference(N+2, N+3)',
            'del(N+2)',
            'count_items(N+3)',
            'count_items(N+2)',
            'del(N+2)',
            'del(N+2)',
            'equal(N+2, 1)',
            'equal(N+2, 3)',
            'del(N+2)',
            'equal(N+2, 4)',
            'del(N+2)',
            'and(N+3, N+4)',
            'del(N+3)',
            'del(N+3)',
            'or(N+2, N+3)',
            'del(N+2)',
            'del(N+2)',
            'switch(N+2, "param1", N+0.c)',
            'del(N+2)',
            'set_color(N+0, N+2)',
            'del(N+2)',
            'del(N+0)',
            'rebuild_grid(N+0, N+1)',
            'del(N+0)',
            'del(N+0)'
        ]

        uuid_prog = ProgUtils.map_refIDs_to_uuids(prog_example)

    expected_uuid_prog = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'del(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(972a8469-1641-4f82-8b9d-2434e465e150)',
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)',
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)',
        'del(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3)',
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a6a4-8b81-48f6-b38a-088ca65ed389, 4)',
        'del(6b65a6a4-8b81-48f6-b38a-088ca65ed389)',
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        'del(c241330b-01a9-471f-9e8a-774bcf36d58b)',
        'del(6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)',
        'del(47378190-96da-4dac-b2ff-5d2a386ecbe0)',
        'del(371ecd7b-27cd-4130-8722-9389571aa876)',
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)',
        'del(1a2a73ed-562b-4f79-8374-59eef50bea63)',
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b8c1e9-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)',
        'del(5be6128e-18c2-4797-a142-ea7d17be3111)',
        'del(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)',
        'del(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9)',
        'del(43b7a3a6-9a8d-4a03-980d-7b71d8f56413)'
    ]

    assert uuid_prog == expected_uuid_prog, "uuid_prog does not match expected_uuid_prog"

def test_map_uuids_to_refIDs():
    uuid_prog = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'del(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(972a8469-1641-4f82-8b9d-2434e465e150)',
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)',
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)',
        'del(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3)',
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a6a4-8b81-48f6-b38a-088ca65ed389, 4)',
        'del(6b65a6a4-8b81-48f6-b38a-088ca65ed389)',
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        'del(c241330b-01a9-471f-9e8a-774bcf36d58b)',
        'del(6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)',
        'del(47378190-96da-4dac-b2ff-5d2a386ecbe0)',
        'del(371ecd7b-27cd-4130-8722-9389571aa876)',
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)',
        'del(1a2a73ed-562b-4f79-8374-59eef50bea63)',
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b8c1e9-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)',
        'del(5be6128e-18c2-4797-a142-ea7d17be3111)',
        'del(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)',
        'del(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9)',
        'del(43b7a3a6-9a8d-4a03-980d-7b71d8f56413)'
    ]

    refID_prog = ProgUtils.map_uuids_to_refIDs(uuid_prog)

    expected_refID_prog = [
        'get_objects(N+0)',
        'get_bg(N+0)',
        'del(N+0)',
        'neighbours8(N+0)',
        'neighbours4(N+0)',
        'set_difference(N+2, N+3)',
        'del(N+2)',
        'count_items(N+3)',
        'count_items(N+2)',
        'del(N+2)',
        'del(N+2)',
        'equal(N+2, 1)',
        'equal(N+2, 3)',
        'del(N+2)',
        'equal(N+2, 4)',
        'del(N+2)',
        'and(N+3, N+4)',
        'del(N+3)',
        'del(N+3)',
        'or(N+2, N+3)',
        'del(N+2)',
        'del(N+2)',
        'switch(N+2, "param1", N+0.c)',
        'del(N+2)',
        'set_color(N+0, N+2)',
        'del(N+2)',
        'del(N+0)',
        'rebuild_grid(N+0, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    assert refID_prog == expected_refID_prog, "refID_prog does not match expected_refID_prog"

def test_remove_dels():
    prog_example = [
        'get_objects(N+0)',
        'get_bg(N+0)',
        'del(N+0)',
        'neighbours8(N+0)',
        'neighbours4(N+0)',
        'set_difference(N+2, N+3)',
        'del(N+2)',
        'count_items(N+3)',
        'count_items(N+2)',
        'del(N+2)',
        'del(N+2)',
        'equal(N+2, 1)',
        'equal(N+2, 3)',
        'del(N+2)',
        'equal(N+2, 4)',
        'del(N+2)',
        'and(N+3, N+4)',
        'del(N+3)',
        'del(N+3)',
        'or(N+2, N+3)',
        'del(N+2)',
        'del(N+2)',
        'switch(N+2, "param1", N+0.c)',
        'del(N+2)',
        'set_color(N+0, N+2)',
        'del(N+2)',
        'del(N+0)',
        'rebuild_grid(N+0, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    prog_without_dels = ProgUtils.remove_dels(prog_example)

    expected_output = [
        'get_objects(N+0)', 
        'get_bg(N+0)', 
        'neighbours8(N+0)', 
        'neighbours4(N+0)', 
        'set_difference(N+2, N+3)', 
        'count_items(N+3)', 
        'count_items(N+2)', 
        'equal(N+2, 1)', 
        'equal(N+2, 3)', 
        'equal(N+2, 4)', 
        'and(N+3, N+4)', 
        'or(N+2, N+3)', 
        'switch(N+2, "param1", N+0.c)', 
        'set_color(N+0, N+2)', 
        'rebuild_grid(N+0, N+1)'
    ]

    assert prog_without_dels == expected_output, "prog_without_dels does not match expected_output"

    uuid_prog = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'del(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(972a8469-1641-4f82-8b9d-2434e465e150)',
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)',
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)',
        'del(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3)',
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a6a4-8b81-48f6-b38a-088ca65ed389, 4)',
        'del(6b65a6a4-8b81-48f6-b38a-088ca65ed389)',
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        'del(c241330b-01a9-471f-9e8a-774bcf36d58b)',
        'del(6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)',
        'del(47378190-96da-4dac-b2ff-5d2a386ecbe0)',
        'del(371ecd7b-27cd-4130-8722-9389571aa876)',
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)',
        'del(1a2a73ed-562b-4f79-8374-59eef50bea63)',
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b8c1e9-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)',
        'del(5be6128e-18c2-4797-a142-ea7d17be3111)',
        'del(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)',
        'del(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9)',
        'del(43b7a3a6-9a8d-4a03-980d-7b71d8f56413)'
    ]

    uuid_prog_without_dels = ProgUtils.remove_dels(uuid_prog)

    expected_output = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)', 
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)', 
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)', 
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)', 
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)', 
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)', 
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)', 
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)', 
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)', 
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a6a4-8b81-48f6-b38a-088ca65ed389, 4)', 
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)', 
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)', 
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)', 
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b8c1e9-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)', 
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)'
    ]

    assert uuid_prog_without_dels == expected_output, "uuid_prog_without_dels does not match expected_output"

    assert(len(uuid_prog_without_dels) == len(prog_without_dels))

def test_auto_add_dels():

    no_del_uuid_prog = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)', 
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)', 
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)', 
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)', 
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)', 
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)', 
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)', 
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)', 
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)', 
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a6a4-8b81-48f6-b38a-088ca65ed389, 4)', 
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)', 
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)', 
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)', 
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b8c1e9-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)', 
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)'
    ]

    added_dels_prog = ProgUtils.auto_add_dels(no_del_uuid_prog)

    expected_uuid_prog = [
        '23b8c1e9-3924-46de-beb1-3b9046685257 = get_objects(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9 = get_bg(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        'del(bdd640fb-0667-4ad1-9c80-317fa3b1799d)',
        '972a8469-1641-4f82-8b9d-2434e465e150 = neighbours8(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '17fc695a-07a0-4a6e-8822-e8f36c031199 = neighbours4(23b8c1e9-3924-46de-beb1-3b9046685257)',
        '9a1de644-815e-46d1-bb8f-aa1837f8a88b = set_difference(972a8469-1641-4f82-8b9d-2434e465e150, 17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(972a8469-1641-4f82-8b9d-2434e465e150)',
        'b74d0fb1-32e7-4629-8fad-c1a606cb0fb3 = count_items(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        'del(9a1de644-815e-46d1-bb8f-aa1837f8a88b)',
        '6b65a6a4-8b81-48f6-b38a-088ca65ed389 = count_items(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        'del(17fc695a-07a0-4a6e-8822-e8f36c031199)',
        '47378190-96da-4dac-b2ff-5d2a386ecbe0 = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 1)',
        'c241330b-01a9-471f-9e8a-774bcf36d58b = equal(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3, 3)',
        'del(b74d0fb1-32e7-4629-8fad-c1a606cb0fb3)',
        '6c307511-b2b9-437a-a8df-6ec4ce4a2bbd = equal(6b65a6a4-8b81-48f6-b38a-088ca65ed389, 4)',
        'del(6b65a6a4-8b81-48f6-b38a-088ca65ed389)',
        '371ecd7b-27cd-4130-8722-9389571aa876 = and(c241330b-01a9-471f-9e8a-774bcf36d58b, 6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        'del(c241330b-01a9-471f-9e8a-774bcf36d58b)',
        'del(6c307511-b2b9-437a-a8df-6ec4ce4a2bbd)',
        '1a2a73ed-562b-4f79-8374-59eef50bea63 = or(47378190-96da-4dac-b2ff-5d2a386ecbe0, 371ecd7b-27cd-4130-8722-9389571aa876)',
        'del(47378190-96da-4dac-b2ff-5d2a386ecbe0)',
        'del(371ecd7b-27cd-4130-8722-9389571aa876)',
        '5be6128e-18c2-4797-a142-ea7d17be3111 = switch(1a2a73ed-562b-4f79-8374-59eef50bea63, "param1", 23b8c1e9-3924-46de-beb1-3b9046685257.c)',
        'del(1a2a73ed-562b-4f79-8374-59eef50bea63)',
        '43b7a3a6-9a8d-4a03-980d-7b71d8f56413 = set_color(23b8c1e9-3924-46de-beb1-3b9046685257, 5be6128e-18c2-4797-a142-ea7d17be3111)',
        'del(23b8c1e9-3924-46de-beb1-3b9046685257)',
        'del(5be6128e-18c2-4797-a142-ea7d17be3111)',
        '759cde66-bacf-43d0-8b1f-9163ce9ff57f = rebuild_grid(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9, 43b7a3a6-9a8d-4a03-980d-7b71d8f56413)',
        'del(43b7a3a6-9a8d-4a03-980d-7b71d8f56413)',
        'del(bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9)'
    ]

    assert added_dels_prog == expected_uuid_prog, "added_dels_prog does not match expected_uuid_prog"

def test_infer_type_new_grid():
    primitive_name = 'new_grid'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS
            
    width_const = 3 + ProgUtils.NUM_SPECIAL_TOKENS
    height_const = 3 + ProgUtils.NUM_SPECIAL_TOKENS
    color_const = 0 + ProgUtils.NUM_SPECIAL_TOKENS

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        width_const,
        ProgUtils.ARG_SEP_TOKEN,
        height_const,
        ProgUtils.ARG_SEP_TOKEN,
        color_const,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = []
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type is TYPE_GRIDOBJECT (since new_grid returns a GridObject)
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"

def test_infer_type_unique():
    primitive_name = 'unique'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    ref_id = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        ref_id,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

def test_infer_type_get_index():
    primitive_name = 'index'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    ref_id = 61
    idx = 5 + ProgUtils.NUM_SPECIAL_TOKENS

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        ref_id,
        ProgUtils.ARG_SEP_TOKEN,
        idx,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_BOOL, \
        f"Expected TYPE_BOOL ({ProgUtils.TYPE_BOOL}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"

def test_infer_type_less_than():
    primitive_name = 'less_than'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    a = 5
    b = 6

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = []
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_BOOL, \
        f"Expected TYPE_BOOL ({ProgUtils.TYPE_BOOL}), got {result_type}"

    a = 5
    b = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_BOOL, \
        f"Expected TYPE_LIST_BOOL ({ProgUtils.TYPE_LIST_BOOL}), got {result_type}"


    a = 62
    b = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_BOOL, \
        f"Expected TYPE_LIST_BOOL ({ProgUtils.TYPE_LIST_BOOL}), got {result_type}"


def test_infer_type_count_items():
    primitive_name = 'count_items'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    data_list = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        data_list,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_PIXEL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"


def test_infer_type_count_values():
    primitive_name = 'count_values'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    values = 61
    data_list = 62

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        values,
        ProgUtils.ARG_SEP_TOKEN,
        data_list,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_GRIDOBJECT]

    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)

    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"


def test_infer_type_sin_half_pi():
    primitive_name = 'sin_half_pi'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    val = 5

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        val,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = []
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    val = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        val,
        ProgUtils.EOS_TOKEN
    ]

    state_var_types = [ProgUtils.TYPE_LIST_INT]

    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)

    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

def test_infer_type_addition():
    primitive_name = 'add'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    a = 5
    b = 5

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN

    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = []
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    a = 61
    b = 62

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN

    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

def test_infer_type_switch():
    primitive_name = 'switch'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    cond1 = 61
    cond2 = 62
    op1 = 63
    op2 = 64
    otherwise = 5

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        cond1,
        ProgUtils.ARG_SEP_TOKEN,
        cond2,
        ProgUtils.ARG_SEP_TOKEN,
        op1,
        ProgUtils.ARG_SEP_TOKEN,
        op2,
        ProgUtils.ARG_SEP_TOKEN,
        otherwise,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

    cond1 = 61
    cond2 = 62
    cond3 = 63
    op1 = 64
    op2 = 65
    op3 = 66
    otherwise = 5

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        cond1,
        ProgUtils.ARG_SEP_TOKEN,
        cond2,
        ProgUtils.ARG_SEP_TOKEN,
        cond3,
        ProgUtils.ARG_SEP_TOKEN,
        op1,
        ProgUtils.ARG_SEP_TOKEN,
        op2,
        ProgUtils.ARG_SEP_TOKEN,
        op3,
        ProgUtils.ARG_SEP_TOKEN,
        otherwise,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

def test_infer_type_keep():
    primitive_name = 'keep'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    data_list = 61
    flags = 62

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        data_list,
        ProgUtils.ARG_SEP_TOKEN,
        flags,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT, ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"


def test_infer_type_set_difference():
    primitive_name = 'set_difference'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    a = 61
    b = 62

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_PIXEL, ProgUtils.TYPE_LIST_PIXEL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_PIXEL, \
        f"Expected TYPE_LIST_PIXEL ({ProgUtils.TYPE_LIST_PIXEL}), got {result_type}"


def test_infer_type_set_pixels():
    primitive_name = 'set_pixels'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    grid = 61
    x = 62
    y = 63
    c = 64

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        grid,
        ProgUtils.ARG_SEP_TOKEN,
        x,
        ProgUtils.ARG_SEP_TOKEN,
        y,
        ProgUtils.ARG_SEP_TOKEN,
        c,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_GRIDOBJECT, ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"


def test_infer_type_crop():
    primitive_name = 'crop'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    grid = 61
    x1 = 4
    y1 = 4
    x2 = 10
    y2 = 10

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        grid,
        ProgUtils.ARG_SEP_TOKEN,
        x1,
        ProgUtils.ARG_SEP_TOKEN,
        y1,
        ProgUtils.ARG_SEP_TOKEN,
        x2,
        ProgUtils.ARG_SEP_TOKEN,
        y2,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"
